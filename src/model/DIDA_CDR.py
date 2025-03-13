import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.autograd import Variable
import numpy as np
import math
from model.GCN import GCN


class DIDA_CDR(nn.Module):
    def __init__(self, opt, source_UV, source_VU, target_UV, target_VU):
        super(DIDA_CDR, self).__init__()
        self.opt=opt
        self.source_UV = source_UV
        self.source_VU = source_VU
        self.target_UV = target_UV
        self.target_VU = target_VU
        self.n_layers = opt['GNN']
        self.dropout = opt['dropout']
        self.emb_size = opt['feature_dim']

        self.dropout = opt["dropout"]
        self.user_num = opt['source_user_num']
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])

        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        self.gcn_s = GCNEncoder(opt)
        self.gcn_t = GCNEncoder(opt)

        self.user_index = self.user_index.cuda()
        self.source_user_index = self.source_user_index.cuda()
        self.target_user_index = self.target_user_index.cuda()
        self.source_item_index = self.source_item_index.cuda()
        self.target_item_index = self.target_item_index.cuda()

        self.att1_s = nn.Sequential(nn.Linear((self.emb_size//2) * 3, self.emb_size//2))
        self.att2_s = nn.Linear(self.emb_size//2, 3)
        self.att1_t = nn.Sequential(nn.Linear((self.emb_size//2) * 3, self.emb_size//2))
        self.att2_t = nn.Linear(self.emb_size//2, 3)

        self.disen_s = Disentangle(opt)
        self.disen_t = Disentangle(opt)
        self.disen_sha = Disentangle(opt)
        self.cls_1 = nn.Sequential(nn.Linear(opt["feature_dim"]//2, 2), nn.Sigmoid())
        self.cls_2 = nn.Sequential(nn.Linear(opt["feature_dim"]//2, 2), nn.Sigmoid())

        self.bcewl = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_KLD = nn.KLDivLoss(reduction='batchmean')
        self.criterion = nn.BCEWithLogitsLoss()

        self.mlp_user_s = nn.Sequential(nn.Linear(opt["feature_dim"]//2, opt["feature_dim"]), 
                                        nn.LeakyReLU(opt['leaky']), 
                                        nn.Linear(opt["feature_dim"], opt["feature_dim"]//2))
        self.mlp_item_s = nn.Sequential(nn.Linear(opt["feature_dim"], opt["feature_dim"]), 
                                        nn.LeakyReLU(opt['leaky']), 
                                        nn.Linear(opt["feature_dim"], opt["feature_dim"]//2))
        
        self.mlp_user_t = nn.Sequential(nn.Linear(opt["feature_dim"]//2, opt["feature_dim"]), 
                                        nn.LeakyReLU(opt['leaky']), 
                                        nn.Linear(opt["feature_dim"], opt["feature_dim"]//2))
        self.mlp_item_t = nn.Sequential(nn.Linear(opt["feature_dim"], opt["feature_dim"]), 
                                        nn.LeakyReLU(opt['leaky']), 
                                        nn.Linear(opt["feature_dim"], opt["feature_dim"]//2))

        # restore graph emb for val and test
        self.restore_user_s = None
        self.restore_item_s = None
        self.restore_user_t = None
        self.restore_item_t = None

        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.source_user_embedding.weight.data)
        xavier_normal_(self.target_user_embedding.weight.data)
        xavier_normal_(self.source_item_embedding.weight.data)
        xavier_normal_(self.target_item_embedding.weight.data)

    def predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
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
    
    def get_G_emb(self):
        source_user = self.source_user_embedding(self.user_index)
        target_user = self.target_user_embedding(self.user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        user_G_emb_s, item_G_emb_s = self.gcn_s(source_user, source_item, self.source_UV, self.source_VU)
        user_G_emb_t, item_G_emb_t = self.gcn_t(target_user, target_item, self.target_UV, self.target_VU)
        return user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t
    
    def mixup_data(self, x_1, x_2, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x_1.size()[0]
        mixed_x = lam * x_1 + (1 - lam) * x_2

        return mixed_x, lam

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
            source_pop_item, target_pop_item):
        if self.restore_user_s is not None or self.restore_item_s is not None:
            self.restore_user_s, self.restore_item_s = None, None
        if self.restore_user_t is not None or self.restore_item_t is not None:
            self.restore_user_t, self.restore_item_t = None, None
            self.restore_val_user_s, self.restore_val_item_s = None, None
        
        user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t  = self.get_G_emb()

        source_user_feature_spe = user_G_emb_s[user]
        target_user_feature_spe = user_G_emb_t[user]

        source_item_pos_feature_spe = item_G_emb_s[source_pos_item]
        target_item_pos_feature_spe = item_G_emb_t[target_pos_item]
        source_item_neg_feature_spe = item_G_emb_s[source_neg_item]
        target_item_neg_feature_spe = item_G_emb_t[target_neg_item]

        user_feature_share, lam = self.mixup_data(source_user_feature_spe, target_user_feature_spe, alpha=1.0)

        domain_sha, domain_spe = self.disen_sha(user_feature_share)
        domain_ind_s, domain_spe_s = self.disen_s(source_user_feature_spe)
        domain_ind_t, domain_spe_t = self.disen_t(target_user_feature_spe)

        # domain cls loss
        episode_batch = user.shape[0]
        y_sha = Variable(torch.ones(episode_batch, 2)/2.0).cuda()   #[0,5, 0.5]
        y_spe_s = Variable(torch.ones(episode_batch).long()).cuda()   #[1.0, 1.0]                         
        y_spe_t = Variable(torch.zeros(episode_batch).long()).cuda()  #[0.0, 0.0]

        loss_cls1 = (1/3) * (self.loss_fn(self.cls_1(domain_spe_s), y_spe_s)+self.loss_fn(self.cls_1(domain_spe_t), y_spe_t) + \
                             lam * self.loss_fn(self.cls_1(domain_spe), y_spe_s) + (1-lam) * self.loss_fn(self.cls_1(domain_spe), y_spe_t))
        
        loss_cls2 = (1/3) * (self.loss_KLD(F.log_softmax(self.cls_2(domain_sha), dim=1), y_sha) + \
                                           self.loss_KLD(F.log_softmax(self.cls_2(domain_ind_s), dim=1), y_sha) + \
                                            self.loss_KLD(F.log_softmax(self.cls_2(domain_ind_t), dim=1), y_sha))

        # fuse
        source_user_feature_fused = self.emb_fusion(domain_spe_s, domain_ind_s, domain_sha, domain='source')
        target_user_feature_fused = self.emb_fusion(domain_spe_t, domain_ind_t, domain_sha, domain='target')

        pos_source_score = self.predict_dot(self.mlp_user_s(source_user_feature_fused), self.mlp_item_s(source_item_pos_feature_spe))
        neg_source_score = self.predict_dot(self.mlp_user_s(source_user_feature_fused), self.mlp_item_s(source_item_neg_feature_spe))
        
        pos_target_score = self.predict_dot(self.mlp_user_t(target_user_feature_fused), self.mlp_item_t(target_item_pos_feature_spe))
        neg_target_score = self.predict_dot(self.mlp_user_t(target_user_feature_fused), self.mlp_item_t(target_item_neg_feature_spe))

        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            pos_source_score.size())
        
        pos_labels = pos_labels.cuda()
        neg_labels = neg_labels.cuda()
        
        loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels)

        reg_loss = (1 / 2) * (source_user_feature_fused.norm(2).pow(2) +
                              target_user_feature_fused.norm(2).pow(2) +
                              source_item_pos_feature_spe.norm(2).pow(2) + 
                               target_item_pos_feature_spe.norm(2).pow(2) +
                                source_item_neg_feature_spe.norm(2).pow(2) + 
                                target_item_neg_feature_spe.norm(2).pow(2)
                              ) / float(len(user))
        all_loss = loss  + self.opt['lambda1']*(loss_cls1) + self.opt['lambda2']*(loss_cls2) + self.opt['reg_weight']*reg_loss

        return all_loss
    
    def get_evaluate_embedding(self):
        if self.restore_user_s is None or self.restore_item_s is None:
            self.restore_user_s, self.restore_item_s, self.restore_user_t, self.restore_item_t =  self.get_G_emb()

        source_user_feature_spe = self.restore_user_s[self.user_index]
        target_user_feature_spe = self.restore_user_t[self.user_index]
        source_item_feature_spe = self.restore_item_s[self.source_item_index]
        target_item_feature_spe = self.restore_item_t[self.target_item_index]

        user_feature_share, lam = self.mixup_data(source_user_feature_spe, target_user_feature_spe, alpha=1.0)
        domain_sha, domain_spe = self.disen_sha(user_feature_share)
        domain_ind_s, domain_spe_s = self.disen_s(source_user_feature_spe)
        domain_ind_t, domain_spe_t = self.disen_t(target_user_feature_spe)

        source_user_feature_fused = self.emb_fusion(domain_spe_s, domain_ind_s, domain_sha, domain='source')
        target_user_feature_fused = self.emb_fusion(domain_spe_t, domain_ind_t, domain_sha, domain='target')
        source_user_feature_fused = self.mlp_user_s(source_user_feature_fused)
        target_user_feature_fused = self.mlp_user_t(target_user_feature_fused)

        source_item_feature = self.mlp_item_s(source_item_feature_spe)
        target_item_feature = self.mlp_item_t(target_item_feature_spe)

        return source_user_feature_fused, source_item_feature, target_user_feature_fused, target_item_feature


class GCNEncoder(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(GCNEncoder, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(DGCNLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = [ufea]
        learn_item = [vfea]
        ufea_G = ufea
        vfea_G = vfea
        for layer in self.encoder:
            ufea_G = F.dropout(ufea_G, self.dropout, training=self.training)
            vfea_G = F.dropout(vfea_G, self.dropout, training=self.training)
            ufea_G, vfea_G = layer(ufea_G, vfea_G, UV_adj, VU_adj)
            learn_user.append(ufea_G)
            learn_item.append(vfea_G)
        learn_user_all = torch.stack(learn_user, dim=1)
        learn_user_all = torch.mean(learn_user_all, dim=1)
        learn_item_all = torch.stack(learn_item, dim=1)
        learn_item_all = torch.mean(learn_item_all, dim=1)
        return learn_user_all, learn_item_all
    

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3 = GCN(
            nfeat=opt["feature_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc4 = GCN(
            nfeat=opt["feature_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        return F.relu(User_ho), F.relu(Item_ho)


class Disentangle(nn.Module):
    def __init__(self, opt):
        super(Disentangle, self).__init__()
        #encoder
        self.emb_size = opt['feature_dim']
        self.fc1 = nn.Linear(self.emb_size, self.emb_size // 2)

        self.fc21a = nn.Linear(self.emb_size // 2, self.emb_size // 2)
        self.fc22a = nn.Linear(self.emb_size // 2, self.emb_size // 2)
        self.fc21b = nn.Linear(self.emb_size // 2, self.emb_size // 2)
        self.fc22b = nn.Linear(self.emb_size // 2, self.emb_size // 2)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
       
        # a encoder: domain irrelevant
        a_mean, a_logvar = self.fc21a(h1), self.fc22a(h1)
        
        # b encoder: domain specific
        b_mean, b_logvar = self.fc21b(h1), self.fc22b(h1)
 
        return a_mean, a_logvar, b_mean, b_logvar


    def reparametrize(self, mu,logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def forward(self, x):
        a_mu, a_logvar, b_mu, b_logvar = self.encode(x)
        a_fea = self.reparametrize(a_mu, a_logvar)             # domain-irrelevant  (H1)
        b_fea = self.reparametrize(b_mu, b_logvar)             # domain-specific    (H2)
        return a_fea, b_fea