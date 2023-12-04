import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class CoPD(nn.Module):
    def __init__(self, opt, s_adj, t_adj, cross_adj):
        super(CoPD, self).__init__()
        self.opt=opt
        self.s_adj = s_adj
        self.t_adj = t_adj
        self.cross_adj = cross_adj
        self.n_layers = opt['GNN']
        self.dropout = opt['dropout']
        self.emb_size = opt['feature_dim']
        self.temp = opt['temp']

        self.dropout = opt["dropout"]
        self.user_num = opt['source_user_num']
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], self.emb_size)
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], self.emb_size)
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], self.emb_size)
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], self.emb_size)
        self.share_user_embedding = nn.Embedding(opt["target_item_num"], self.emb_size)

        self.disen_encoder = nn.Sequential(
                                        nn.Linear(self.emb_size, self.emb_size),
                                        nn.Dropout(opt['dropout']),
                                        nn.ReLU(),
                                        nn.Linear(self.emb_size, 2 * self.emb_size))
        self.domain_cls = nn.Sequential(nn.Linear(self.emb_size, 2), nn.Sigmoid())

        self.user_index = torch.arange(0, self.opt["source_user_num"], 1).cuda()
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1).cuda()
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1).cuda()
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1).cuda()
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1).cuda()

        self.att1_s = nn.Sequential(nn.Linear(self.emb_size * 3, self.emb_size))
        self.att2_s = nn.Linear(self.emb_size, 3)
        self.att1_t = nn.Sequential(nn.Linear(self.emb_size * 3, self.emb_size))
        self.att2_t = nn.Linear(self.emb_size, 3)

        self.agg_s = nn.Linear(self.emb_size * 2, self.emb_size)
        self.agg_t = nn.Linear(self.emb_size * 2, self.emb_size)

        self.dis_s = nn.Bilinear(self.emb_size, self.emb_size, 1)
        self.dis_t = nn.Bilinear(self.emb_size, self.emb_size, 1)

        self.loss_cos = nn.CosineEmbeddingLoss()
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_KLD = nn.KLDivLoss(reduction='batchmean')

        # restore graph emb for val and test
        self.restore_user_s = None
        self.restore_item_s = None
        self.restore_user_t = None
        self.restore_item_t = None
        self.restore_user_sha = None
        self.restore_item_sha_s = None
        self.restore_item_sha_t = None
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.source_user_embedding.weight.data)
        xavier_uniform_(self.target_user_embedding.weight.data)
        xavier_uniform_(self.source_item_embedding.weight.data)
        xavier_uniform_(self.target_item_embedding.weight.data)
        xavier_uniform_(self.share_user_embedding.weight.data)

    def predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        return output

    def emb_fusion(self, spe_emb, int_emb, pop_emb, domain='source'):
        if domain == 'source':
            att1 = self.att1_s
            att2 = self.att2_s
        else:
            att1 = self.att1_t
            att2 = self.att2_t
        x = torch.concat([spe_emb, int_emb, pop_emb], dim=1)
        x = F.relu(att1(x).cuda(), inplace = True)
        x = F.dropout(x, training =self.training, p = self.dropout)
        x = att2(x).cuda()
        att_w = F.softmax(x, dim = 1)
        att_w1, att_w2, att_w3 = att_w.chunk(3, dim = 1)
        att_w1.repeat(self.emb_size, 1)
        att_w2.repeat(self.emb_size, 1)
        att_w3.repeat(self.emb_size, 1)
        integrated_emb = torch.mul(spe_emb, att_w1) + torch.mul(int_emb, att_w2) + torch.mul(pop_emb, att_w3)
        return integrated_emb
    
    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat._indices()
        values = nn.functional.dropout(mat._values(), p=dropout)
        size = mat.size()
        return torch.sparse.FloatTensor(indices, values, size)
    
    def get_G_emb(self):
        source_user = self.source_user_embedding(self.user_index)
        target_user = self.target_user_embedding(self.user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        share_user = self.share_user_embedding(self.user_index)

        # return source_user, source_item, target_user, target_item
        all_embeddings_s = torch.cat([source_user, source_item], dim=0)  # (m+n1) * d
        all_embeddings_t = torch.cat([target_user, target_item], dim=0)  # (m+n2) * d

        all_embeddings_share = torch.cat([share_user, source_item, target_item], dim=0)  # (m+n1+n2) * d

        user_G_emb_s, item_G_emb_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num)
        user_G_emb_t, item_G_emb_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num)
        user_G_emb_sha, item_G_emb_sha= self.lightGCN_forward(all_embeddings_share, self.cross_adj, \
                                                     self.user_num, self.s_item_num+self.t_item_num)
        item_G_emb_sha_s, item_G_emg_sha_t = torch.split(item_G_emb_sha, [self.s_item_num, self.t_item_num])
        return user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t, user_G_emb_sha, item_G_emb_sha_s, item_G_emg_sha_t
    
    def lightGCN_forward(self, all_embeddings, norm_adj_matrix, n_users, n_items):
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.sparse_dropout(norm_adj_matrix,self.dropout), all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [n_users, n_items])

        return user_all_embeddings, item_all_embeddings

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
            source_pop_item, target_pop_item):
        if self.restore_user_s is not None or self.restore_item_s is not None:
            self.restore_user_s, self.restore_item_s = None, None
        
        user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t, \
            user_G_emb_sha, item_G_emb_sha_s, item_G_emb_sha_t = self.get_G_emb()

        source_user_feature_spe = user_G_emb_s[user]
        target_user_feature_spe = user_G_emb_t[user]
        user_feature_share = user_G_emb_sha[user]

        source_item_pos_feature_spe = item_G_emb_s[source_pos_item]
        target_item_pos_feature_spe = item_G_emb_t[target_pos_item]
        source_item_neg_feature_spe = item_G_emb_s[source_neg_item]
        target_item_neg_feature_spe = item_G_emb_t[target_neg_item]
        source_item_pos_feature_share = item_G_emb_sha_s[source_pos_item]
        target_item_pos_feature_share = item_G_emb_sha_t[target_pos_item]
        source_item_neg_feature_share = item_G_emb_sha_s[source_neg_item]
        target_item_neg_feature_share = item_G_emb_sha_t[target_neg_item]

        episode_batch = user.shape[0]
        # 1.Consistency constraint loss
        pos_label = torch.ones(episode_batch).long().cuda() # torch.Size(batch_size)
        neg_label = -torch.ones(episode_batch).long().cuda()
        loss_cc_A = self.loss_cos(source_item_pos_feature_spe.cpu().detach().cuda(), source_item_pos_feature_share, pos_label) + \
            self.loss_cos(source_item_neg_feature_spe.cpu().detach().cuda(), source_item_pos_feature_share, neg_label)
        loss_cc_B = self.loss_cos(target_item_pos_feature_spe.cpu().detach().cuda(), target_item_pos_feature_share, pos_label) + \
            self.loss_cos(target_item_neg_feature_spe.cpu().detach().cuda(), target_item_pos_feature_share, neg_label)
        loss_cc = loss_cc_A + loss_cc_B

        # 2.Domain classifier loss of user preference embeddings
        y_S = (torch.ones(episode_batch, 2)/2.0).cuda()   #[0,5, 0.5]
        y_A = torch.ones(episode_batch).long().cuda()                         
        y_B = torch.zeros(episode_batch).long().cuda()

        source_spe_score = self.domain_cls(source_user_feature_spe)
        target_spe_score = self.domain_cls(target_user_feature_spe)
        share_score = self.domain_cls(user_feature_share)
        loss_share_kld = self.loss_KLD(F.log_softmax(share_score, dim=1), y_S)
        loss_domain_CLS_A = self.loss_CE(source_spe_score, y_A) 
        loss_domain_CLS_B = self.loss_CE(target_spe_score, y_B)
        loss_dom = loss_share_kld + loss_domain_CLS_A + loss_domain_CLS_B
        
        # 3.Preference disentanglement loss
        int_user_feature, pop_user_feature = torch.split(self.disen_encoder(user_feature_share), \
                                                    split_size_or_sections=self.emb_size, dim=1)
        pos_source_int_score = self.predict_dot(int_user_feature, source_item_pos_feature_share)
        pos_target_int_score = self.predict_dot(int_user_feature, target_item_pos_feature_share)
        pos_source_pop_score = self.predict_dot(pop_user_feature, source_item_pos_feature_share)
        pos_target_pop_score = self.predict_dot(pop_user_feature, target_item_pos_feature_share)

        conf_weight_A = torch.exp(source_pop_item.unsqueeze(1))
        conf_weight_B = torch.exp(target_pop_item.unsqueeze(1))
        int_weight_A = torch.exp(torch.ones_like(conf_weight_A)-source_pop_item.unsqueeze(1))
        int_weight_B = torch.exp(torch.ones_like(conf_weight_B)-target_pop_item.unsqueeze(1))

        loss_conf_A = -torch.log(conf_weight_A * torch.exp(pos_source_pop_score/self.temp)).mean() + \
            torch.log(torch.exp(pop_user_feature @ item_G_emb_sha_s.T / self.temp).sum(1) + 1e-8).mean()
        loss_conf_B = - torch.log(conf_weight_B * torch.exp(pos_target_pop_score/self.temp)).mean() +\
            torch.log(torch.exp(pop_user_feature @ item_G_emb_sha_t.T / self.temp).sum(1) + 1e-8).mean()
        loss_int_A = -torch.log(int_weight_A * torch.exp(pos_source_int_score/self.temp)).mean() + \
            torch.log(torch.exp(int_user_feature @ item_G_emb_sha_s.T / self.temp).sum(1) + 1e-8).mean()
        loss_int_B = - torch.log(int_weight_B * torch.exp(pos_target_int_score/self.temp)).mean() + \
            torch.log(torch.exp(int_user_feature @ item_G_emb_sha_t.T / self.temp).sum(1) + 1e-8).mean()
        loss_pd = loss_conf_A + loss_conf_B + loss_int_A + loss_int_B

        # 4.Emb fusion and Recommendation
        source_user_feature_fused = self.emb_fusion(source_user_feature_spe, int_user_feature, pop_user_feature, domain='source')
        target_user_feature_fused = self.emb_fusion(target_user_feature_spe, int_user_feature, pop_user_feature, domain='target')

        source_item_pos_feature_fused = self.agg_s(torch.cat([source_item_pos_feature_spe, source_item_pos_feature_share], dim=1))
        source_item_neg_feature_fused = self.agg_s(torch.cat([source_item_neg_feature_spe, source_item_neg_feature_share], dim=1))
        target_item_pos_feature_fused = self.agg_t(torch.cat([target_item_pos_feature_spe, target_item_pos_feature_share], dim=1))
        target_item_neg_feature_fused = self.agg_t(torch.cat([target_item_neg_feature_spe, target_item_neg_feature_share], dim=1))

        pos_source_score = self.predict_dot(source_user_feature_fused, source_item_pos_feature_fused)
        neg_source_score = self.predict_dot(source_user_feature_fused, source_item_neg_feature_fused)
        
        pos_target_score = self.predict_dot(target_user_feature_fused, target_item_pos_feature_fused)
        neg_target_score = self.predict_dot(target_user_feature_fused, target_item_neg_feature_fused)

        loss_bpr_source = torch.mean(torch.nn.functional.softplus(neg_source_score - pos_source_score)) 
        loss_bpr_target = torch.mean(torch.nn.functional.softplus(neg_target_score - pos_target_score))

        # get ego_emb for emb regular
        source_user_feature_ego = self.source_user_embedding(user)
        target_user_feature_ego = self.target_user_embedding(user)
        source_item_pos_feature_ego = self.source_item_embedding(source_pos_item)
        target_item_pos_feature_ego = self.target_item_embedding(target_pos_item)
        source_item_neg_feature_ego = self.source_item_embedding(source_neg_item)
        target_item_neg_feature_ego = self.target_item_embedding(target_neg_item)
        user_feature_share_ego = self.share_user_embedding(user)
        # emb 二范数Loss
        reg_loss = (1 / 2) * (source_user_feature_ego.norm(2).pow(2) +
                              target_user_feature_ego.norm(2).pow(2) +
                              source_item_pos_feature_ego.norm(2).pow(2) + 
                               target_item_pos_feature_ego.norm(2).pow(2) +
                                source_item_neg_feature_ego.norm(2).pow(2) + 
                                target_item_neg_feature_ego.norm(2).pow(2) +
                                user_feature_share_ego.norm(2).pow(2)
                              ) / float(len(user))
        loss_rec = loss_bpr_source + loss_bpr_target + self.opt['reg_weight']*reg_loss

        loss = loss_rec + loss_dom + self.opt['lambda1']*loss_cc + self.opt['lambda2']*loss_pd
        return loss
    
    def get_evaluate_embedding(self):
        if self.restore_user_s is None or self.restore_item_s is None:
            self.restore_user_s, self.restore_item_s, self.restore_user_t, self.restore_item_t, self.restore_user_sha, \
                self.restore_item_sha_s, self.restore_item_sha_t =  self.get_G_emb()

        source_user_feature_spe = self.restore_user_s[self.user_index]
        target_user_feature_spe = self.restore_user_t[self.user_index]
        source_item_feature_spe = self.restore_item_s[self.source_item_index]
        target_item_feature_spe = self.restore_item_t[self.target_item_index]

        user_feature_share = self.restore_user_sha[self.user_index]
        source_item_feature_share = self.restore_item_sha_s[self.source_item_index]
        target_item_feature_share = self.restore_item_sha_t[self.target_item_index]

        int_user_feature, pop_user_feature = torch.split(self.disen_encoder(user_feature_share) , \
                                                         split_size_or_sections=self.emb_size, dim=1)
        
        source_user_feature_fused = self.emb_fusion(source_user_feature_spe, int_user_feature, pop_user_feature, domain='source')
        target_user_feature_fused = self.emb_fusion(target_user_feature_spe, int_user_feature, pop_user_feature, domain='target')

        source_item_feature_fused = self.agg_s(torch.cat([source_item_feature_spe, source_item_feature_share],dim=1))
        target_item_feature_fused = self.agg_t(torch.cat([target_item_feature_spe, target_item_feature_share],dim=1))

        return source_user_feature_fused, source_item_feature_fused, target_user_feature_fused, target_item_feature_fused

