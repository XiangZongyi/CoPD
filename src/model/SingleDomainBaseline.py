import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.autograd import Variable
import random as rd

class DGCF(nn.Module):
    def __init__(self, opt, source_UV, target_UV, s_adj, t_adj):
        super(DGCF, self).__init__()
        self.opt=opt
        self.n_layers = 1
        self.dropout = opt['dropout']
        self.emb_size = opt['feature_dim']
        self.s_adj = s_adj
        self.t_adj = t_adj

        self.user_num = opt['source_user_num']
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        # DGCF需要调的参数
        self.reg_weight = 1e-4
        self.n_factors = 4
        self.n_iterations = 2
        self.cor_weight = 0.01
        n_batch_s = opt["inter_num_s"] // opt['batch_size'] + 1
        self.cor_batch_size_s = int(max(self.user_num / n_batch_s, self.s_item_num / n_batch_s))
        n_batch_t = opt["inter_num_t"] // opt['batch_size'] + 1
        self.cor_batch_size_t = int(max(self.user_num / n_batch_t, self.t_item_num / n_batch_t))
        # ensure embedding can be divided into <n_factors> intent
        assert self.emb_size % self.n_factors == 0
        # source domain
        row = source_UV.row.tolist()
        col = source_UV.col.tolist()
        col = [item_index + self.user_num for item_index in col]
        all_h_list = row + col  # row.extend(col)
        all_t_list = col + row  # col.extend(row)
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list_s = torch.LongTensor(all_h_list).cuda()
        self.all_t_list_s = torch.LongTensor(all_t_list).cuda()
        self.edge2head_s = torch.LongTensor([all_h_list, edge_ids]).cuda()
        self.head2edge_s = torch.LongTensor([edge_ids, all_h_list]).cuda()
        self.tail2edge_s = torch.LongTensor([edge_ids, all_t_list]).cuda()
        val_one = torch.ones_like(self.all_h_list_s).float().cuda()
        num_node = self.user_num + self.s_item_num
        self.edge2head_mat_s = self._build_sparse_tensor(
            self.edge2head_s, val_one, (num_node, num_edge)
        )
        self.head2edge_mat_s = self._build_sparse_tensor(
            self.head2edge_s, val_one, (num_edge, num_node)
        )
        self.tail2edge_mat_s = self._build_sparse_tensor(
            self.tail2edge_s, val_one, (num_edge, num_node)
        )
        self.num_edge_s = num_edge
        self.num_node_s = num_node
        # target_domain
        row = target_UV.row.tolist()
        col = target_UV.col.tolist()
        col = [item_index + self.user_num for item_index in col]
        all_h_list = row + col  # row.extend(col)
        all_t_list = col + row  # col.extend(row)
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list_t = torch.LongTensor(all_h_list).cuda()
        self.all_t_list_t = torch.LongTensor(all_t_list).cuda()
        self.edge2head_t = torch.LongTensor([all_h_list, edge_ids]).cuda()
        self.head2edge_t = torch.LongTensor([edge_ids, all_h_list]).cuda()
        self.tail2edge_t = torch.LongTensor([edge_ids, all_t_list]).cuda()
        val_one = torch.ones_like(self.all_h_list_t).float().cuda()
        num_node = self.user_num + self.t_item_num
        self.edge2head_mat_t = self._build_sparse_tensor(
            self.edge2head_t, val_one, (num_node, num_edge)
        )
        self.head2edge_mat_t = self._build_sparse_tensor(
            self.head2edge_t, val_one, (num_edge, num_node)
        )
        self.tail2edge_mat_t = self._build_sparse_tensor(
            self.tail2edge_t, val_one, (num_edge, num_node)
        )
        self.num_edge_t = num_edge
        self.num_node_t = num_node


        self.source_user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.target_user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.source_item_embedding = nn.Embedding(self.s_item_num, self.emb_size)
        self.target_item_embedding = nn.Embedding(self.t_item_num, self.emb_size)

        self.softmax = torch.nn.Softmax(dim=1)

        self.user_index = torch.arange(0, self.user_num, 1).cuda()
        self.source_user_index = torch.arange(0, self.user_num, 1).cuda()
        self.target_user_index = torch.arange(0, self.user_num, 1).cuda()
        self.source_item_index = torch.arange(0, self.s_item_num , 1).cuda()
        self.target_item_index = torch.arange(0, self.t_item_num, 1).cuda()

        # restore graph emb for val and test
        self.restore_user_s = None
        self.restore_item_s = None
        self.restore_user_t = None
        self.restore_item_t = None
        self._init_weights()

    def _init_weights(self):
        xavier_normal_(self.source_user_embedding.weight.data)
        xavier_uniform_(self.target_user_embedding.weight.data)
        xavier_uniform_(self.source_item_embedding.weight.data)
        xavier_uniform_(self.target_item_embedding.weight.data)
    
    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse.FloatTensor(indices, values, size).cuda()
    
    def build_matrix(self, A_values, source=True):
        r"""Get the normalized interaction matrix of users and items according to A_values.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        Args:
            A_values (torch.cuda.FloatTensor): (num_edge, n_factors)

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            torch.cuda.FloatTensor: Sparse tensor of the normalized interaction matrix. shape: (num_edge, n_factors)
        """
        norm_A_values = self.softmax(A_values)
        factor_edge_weight = []
        for i in range(self.n_factors):
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # (num_edge, 1)
            if source:
                d_values = torch.sparse.mm(self.edge2head_mat_s, tp_values)
            else:
                d_values = torch.sparse.mm(self.edge2head_mat_t, tp_values)
            # (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                print("d_values", torch.min(d_values), torch.max(d_values))

            d_values = 1.0 / torch.sqrt(d_values)
            if source:
                head_term = torch.sparse.mm(self.head2edge_mat_s, d_values)
                tail_term = torch.sparse.mm(self.tail2edge_mat_s, d_values)
            else:
                head_term = torch.sparse.mm(self.head2edge_mat_t, d_values)
                tail_term = torch.sparse.mm(self.tail2edge_mat_t, d_values)
            # (num_edge, num_node) (num_node, 1) -> (num_edge, 1)

            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight
    
    def get_G_emb(self):
        source_user = self.source_user_embedding(self.user_index)
        target_user = self.target_user_embedding(self.user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        # return source_user, source_item, target_user, target_item
        all_embeddings_s = torch.cat([source_user, source_item], dim=0)  # (m+n1) * d
        all_embeddings_t = torch.cat([target_user, target_item], dim=0)  # (m+n2) * d

        user_G_emb_s, item_G_emb_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num, True)
        user_G_emb_t, item_G_emb_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num, False)

        return user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t
    
    def lightGCN_forward(self, ego_embeddings, norm_adj_matrix, n_users, n_items, source=True):
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        if source:
            A_values = torch.ones((self.num_edge_s, self.n_factors)).cuda()
        else:
            A_values = torch.ones((self.num_edge_t, self.n_factors)).cuda()
        A_values = Variable(A_values, requires_grad=True)
        for _ in range(self.n_layers):
            layer_embeddings = []
            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-length list of embeddings
            # [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.n_factors, 1)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values, source=source)
                for i in range(0, self.n_factors):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # (num_edge, 1)
                    if source:
                        edge_val = torch.sparse.mm(
                            self.tail2edge_mat_s, ego_layer_embeddings[i]
                        )
                    else:
                        edge_val = torch.sparse.mm(
                            self.tail2edge_mat_t, ego_layer_embeddings[i]
                        )
                    # (num_edge, dim / n_factors)
                    edge_val = edge_val * edge_weight
                    # (num_edge, dim / n_factors)
                    if source:
                        factor_embeddings = torch.sparse.mm(self.edge2head_mat_s, edge_val)
                    else:
                        factor_embeddings = torch.sparse.mm(self.edge2head_mat_t, edge_val)
                    # (num_node, num_edge) (num_edge, dim) -> (num_node, dim)

                    iter_embeddings.append(factor_embeddings)

                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    if source:
                        head_factor_embeddings = torch.index_select(
                            factor_embeddings, dim=0, index=self.all_h_list_s
                        )
                        tail_factor_embeddings = torch.index_select(
                            ego_layer_embeddings[i], dim=0, index=self.all_t_list_s
                        )
                    else:
                        head_factor_embeddings = torch.index_select(
                            factor_embeddings, dim=0, index=self.all_h_list_t
                        )
                        tail_factor_embeddings = torch.index_select(
                            ego_layer_embeddings[i], dim=0, index=self.all_t_list_t
                        )

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    # to adapt to torch version
                    head_factor_embeddings = F.normalize(
                        head_factor_embeddings, p=2, dim=1
                    )
                    tail_factor_embeddings = F.normalize(
                        tail_factor_embeddings, p=2, dim=1
                    )

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [num_edge, 1]
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings),
                        dim=1,
                        keepdim=True,
                    )

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # (num_edge, n_factors)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, n_layer + 1, embedding_size)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        # (num_node, embedding_size)

        u_g_embeddings = all_embeddings[:n_users, :]
        i_g_embeddings = all_embeddings[n_users :, :]

        return u_g_embeddings, i_g_embeddings
    
    def sample_cor_samples(self, n_users, n_items, cor_batch_size):
        r"""This is a function that sample item ids and user ids.

        Args:
            n_users (int): number of users in total
            n_items (int): number of items in total
            cor_batch_size (int): number of id to sample

        Returns:
            list: cor_users, cor_items. The result sampled ids with both as cor_batch_size long.

        Note:
            We have to sample some embedded representations out of all nodes.
            Because we have no way to store cor-distance for each pair.
        """
        cor_users = rd.sample(list(range(n_users)), cor_batch_size)
        cor_items = rd.sample(list(range(n_items)), cor_batch_size)

        return cor_users, cor_items
    

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
            source_pop_item, target_pop_item):
        if self.restore_user_s is not None or self.restore_item_s is not None:
            self.restore_user_s, self.restore_item_s = None, None
        
        user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t = self.get_G_emb()   
        
        source_user_feature_spe = user_G_emb_s[user]
        target_user_feature_spe = user_G_emb_t[user]

        source_item_pos_feature_spe = item_G_emb_s[source_pos_item]
        target_item_pos_feature_spe = item_G_emb_t[target_pos_item]
        source_item_neg_feature_spe = item_G_emb_s[source_neg_item]
        target_item_neg_feature_spe = item_G_emb_t[target_neg_item]

        # 1.bpr loss
        pos_source_score = (source_user_feature_spe * source_item_pos_feature_spe).sum(dim=-1)
        neg_source_score = (source_user_feature_spe * source_item_neg_feature_spe).sum(dim=-1)
        
        pos_target_score = (target_user_feature_spe * target_item_pos_feature_spe).sum(dim=-1)
        neg_target_score = (target_user_feature_spe * target_item_neg_feature_spe).sum(dim=-1)

        loss_bpr_source = torch.mean(torch.nn.functional.softplus(neg_source_score - pos_source_score)) 
        loss_bpr_target = torch.mean(torch.nn.functional.softplus(neg_target_score - pos_target_score))

        # get ego_emb for emb regular
        source_user_feature_ego = self.source_user_embedding(user)
        target_user_feature_ego = self.target_user_embedding(user)
        source_item_pos_feature_ego = self.source_item_embedding(source_pos_item)
        target_item_pos_feature_ego = self.target_item_embedding(target_pos_item)
        source_item_neg_feature_ego = self.source_item_embedding(source_neg_item)
        target_item_neg_feature_ego = self.target_item_embedding(target_neg_item)

        # reg_loss = (1 / 2) * (source_user_feature_ego.norm(2).pow(2) +
        #                 target_user_feature_ego.norm(2).pow(2) +
        #                 source_item_pos_feature_ego.norm(2).pow(2) + 
        #                 target_item_pos_feature_ego.norm(2).pow(2) +
        #                 source_item_neg_feature_ego.norm(2).pow(2) + 
        #                 target_item_neg_feature_ego.norm(2).pow(2)
        #                 ) / float(len(user))
        reg_loss1 = (1 / 2) * (source_user_feature_ego.norm(2).pow(2) +
                source_item_pos_feature_ego.norm(2).pow(2) + 
                source_item_neg_feature_ego.norm(2).pow(2)
                ) / float(len(user))
        reg_loss2 = (1 / 2) * (
                target_user_feature_ego.norm(2).pow(2) +
                target_item_pos_feature_ego.norm(2).pow(2) +
                target_item_neg_feature_ego.norm(2).pow(2)
                ) / float(len(user))
        if self.n_factors > 1 and self.cor_weight > 1e-9:
            cor_users_s, cor_items_s = self.sample_cor_samples(
                self.user_num, self.s_item_num, self.cor_batch_size_s
            )
            cor_users_t, cor_items_t = self.sample_cor_samples(
                self.user_num, self.t_item_num, self.cor_batch_size_t
            )
            cor_users_s = torch.LongTensor(cor_users_s).cuda()
            cor_items_s = torch.LongTensor(cor_items_s).cuda()
            cor_users_t = torch.LongTensor(cor_users_t).cuda()
            cor_items_t = torch.LongTensor(cor_items_t).cuda()
            cor_u_embeddings_s = user_G_emb_s[cor_users_s]
            cor_i_embeddings_s = item_G_emb_s[cor_items_s]
            cor_u_embeddings_t = user_G_emb_t[cor_users_t]
            cor_i_embeddings_t = item_G_emb_t[cor_items_t]
            cor_loss_s = self.create_cor_loss(cor_u_embeddings_s, cor_i_embeddings_s)
            cor_loss_t = self.create_cor_loss(cor_u_embeddings_t, cor_i_embeddings_t)
            loss_s = loss_bpr_source + self.reg_weight * reg_loss1 + self.cor_weight * (cor_loss_s)
            loss_t = loss_bpr_target + self.reg_weight * reg_loss2 + self.cor_weight * (cor_loss_t)
        else:
            loss_s = loss_bpr_source + self.reg_weight*reg_loss1
            loss_t = loss_bpr_target + self.reg_weight*reg_loss2
        return loss_s + loss_t
    
    def get_evaluate_embedding(self):
        if self.restore_user_s is None or self.restore_item_s is None:
            self.restore_user_s, self.restore_item_s, self.restore_user_t, self.restore_item_t=  self.get_G_emb()

        source_user_feature_spe = self.restore_user_s[self.user_index]
        target_user_feature_spe = self.restore_user_t[self.user_index]
        source_item_feature_spe = self.restore_item_s[self.source_item_index]
        target_item_feature_spe = self.restore_item_t[self.target_item_index]
      

        return source_user_feature_spe, source_item_feature_spe, target_user_feature_spe, target_item_feature_spe

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        r"""Calculate the correlation loss for a sampled users and items.

        Args:
            cor_u_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)
            cor_i_embeddings (torch.cuda.FloatTensor): (cor_batch_size, n_factors)

        Returns:
            torch.Tensor : correlation loss.
        """
        cor_loss = None

        ui_embeddings = torch.cat((cor_u_embeddings, cor_i_embeddings), dim=0)
        ui_factor_embeddings = torch.chunk(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            # (M + N, emb_size / n_factor)
            y = ui_factor_embeddings[i + 1]
            # (M + N, emb_size / n_factor)
            if i == 0:
                cor_loss = self._create_distance_correlation(x, y)
            else:
                cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= (self.n_factors + 1.0) * self.n_factors / 2

        return cor_loss

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            """
            X: (batch_size, dim)
            return: X - E(X)
            """
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(X * X, dim=1, keepdim=True)
            # (N, 1)
            # (x^2 - 2xy + y^2) -> l2 distance between all vectors
            value = r - 2 * torch.mm(X, X.T) + r.T
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            # matrix - average over row - average over col + average over matrix
            D = (
                D
                - torch.mean(D, dim=0, keepdim=True)
                - torch.mean(D, dim=1, keepdim=True)
                + torch.mean(D)
            )
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor



class DICE(nn.Module):
    def __init__(self, opt, s_adj, t_adj):
        super(DICE, self).__init__()
        self.opt=opt
        self.emb_size = opt['feature_dim'] // 2
        self.s_adj = s_adj
        self.t_adj = t_adj
        self.n_layers = 1
        self.dropout = 0.2

        # DICE的参数
        self.dis_loss = 'L1' 
        self.dis_pen = 0.01
        self.int_weight = 0.1
        self.pop_weight = 0.1

        # refer to https://recbole.io/docs/user_guide/config/training_settings.html
        # train_neg_sample_args:        # (dict) Negative sampling configuration for model training.
        # distribution: popularity      # (str) The distribution of negative items.
        # sample_num: 2                 # (int) The sampled num of negative items.
        # alpha: 1.0                    # (float) The power of sampling probability for popularity distribution.
        # dynamic: False                # (bool) Whether to use dynamic negative sampling.
        # candidate_num: 0              # (int) The number of candidate negative items when dynamic negative sampling.

        if self.dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif self.dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()

        self.user_num = opt['source_user_num']
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        self.source_user_embedding_int = nn.Embedding(self.user_num, self.emb_size)
        self.target_user_embedding_int = nn.Embedding(self.user_num, self.emb_size)
        self.source_item_embedding_int = nn.Embedding(self.s_item_num, self.emb_size)
        self.target_item_embedding_int = nn.Embedding(self.t_item_num, self.emb_size)

        self.source_user_embedding_pop = nn.Embedding(self.user_num, self.emb_size)
        self.target_user_embedding_pop = nn.Embedding(self.user_num, self.emb_size)
        self.source_item_embedding_pop = nn.Embedding(self.s_item_num, self.emb_size)
        self.target_item_embedding_pop = nn.Embedding(self.t_item_num, self.emb_size)

        self.user_index = torch.arange(0, self.user_num, 1).cuda()
        self.source_user_index = torch.arange(0, self.user_num, 1).cuda()
        self.target_user_index = torch.arange(0, self.user_num, 1).cuda()
        self.source_item_index = torch.arange(0, self.s_item_num , 1).cuda()
        self.target_item_index = torch.arange(0, self.t_item_num, 1).cuda()

        self._init_weights()
        self.store_user_s = None
        self.store_item_s = None 
        self.store_user_t = None 
        self.store_item_t = None


    def _init_weights(self):
        xavier_uniform_(self.source_user_embedding_int.weight.data)
        xavier_uniform_(self.target_user_embedding_int.weight.data)
        xavier_uniform_(self.source_item_embedding_int.weight.data)
        xavier_uniform_(self.target_item_embedding_int.weight.data)
        xavier_uniform_(self.source_user_embedding_pop.weight.data)
        xavier_uniform_(self.target_user_embedding_pop.weight.data)
        xavier_uniform_(self.source_item_embedding_pop.weight.data)
        xavier_uniform_(self.target_item_embedding_pop.weight.data)
    
    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat._indices()
        values = nn.functional.dropout(mat._values(), p=dropout)
        size = mat.size()
        return torch.sparse.FloatTensor(indices, values, size)

    def lightGCN_forward(self, all_embeddings, norm_adj_matrix, n_users, n_items):
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.sparse_dropout(norm_adj_matrix,self.dropout), all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [n_users, n_items])
        return user_all_embeddings, item_all_embeddings

    def forward_emb(self, factor):
        if factor == 'int':
            user_emb_s = self.source_user_embedding_int.weight
            item_emb_s = self.source_item_embedding_int.weight
            all_embeddings_s = torch.cat([user_emb_s, item_emb_s], dim=0)
            user_emb_G_s, item_emb_G_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num)
            user_emb_t = self.target_user_embedding_int.weight
            item_emb_t = self.target_item_embedding_int.weight
            all_embeddings_t = torch.cat([user_emb_t, item_emb_t], dim=0)
            user_emb_G_t, item_emb_G_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num)
        elif factor == 'pop':
            user_emb_s = self.source_user_embedding_pop.weight
            item_emb_s = self.source_item_embedding_pop.weight
            user_emb_t = self.target_user_embedding_pop.weight
            item_emb_t = self.target_item_embedding_pop.weight
            all_embeddings_s = torch.cat([user_emb_s, item_emb_s], dim=0)
            user_emb_G_s, item_emb_G_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num)
            all_embeddings_t = torch.cat([user_emb_t, item_emb_t], dim=0)
            user_emb_G_t, item_emb_G_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num)
        elif factor == 'tot':
            user_emb_s = torch.cat(
                (self.source_user_embedding_int.weight, self.source_user_embedding_pop.weight), 1)
            item_emb_s = torch.cat(
                (self.source_item_embedding_int.weight, self.source_item_embedding_pop.weight), 1)
            user_emb_t = torch.cat(
                (self.target_user_embedding_int.weight, self.target_user_embedding_pop.weight), 1)
            item_emb_t = torch.cat(
                (self.target_item_embedding_int.weight, self.target_item_embedding_pop.weight), 1)
            all_embeddings_s = torch.cat([user_emb_s, item_emb_s], dim=0)
            user_emb_G_s, item_emb_G_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num)
            all_embeddings_t = torch.cat([user_emb_t, item_emb_t], dim=0)
            user_emb_G_t, item_emb_G_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num)
        return user_emb_G_s, item_emb_G_s, user_emb_G_t, item_emb_G_t

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, mask_s, mask_t):
        if self.store_user_s is not None:
            self.store_user_s = None
            self.store_item_s = None 
            self.store_user_t = None 
            self.store_item_t = None

        user_emb_G_s_int, item_emb_G_s_int, user_emb_G_t_int, item_emb_G_t_int = self.forward_emb('int')
        user_emb_G_s_pop, item_emb_G_s_pop, user_emb_G_t_pop, item_emb_G_t_pop = self.forward_emb('pop')
        
        score_p_int_s = (user_emb_G_s_int[user] * item_emb_G_s_int[source_pos_item]).sum(dim=1)
        score_n_int_s = (user_emb_G_s_int[user] * item_emb_G_s_int[source_neg_item]).sum(dim=1)
        score_p_pop_s = (user_emb_G_s_pop[user] * item_emb_G_s_pop[source_pos_item]).sum(dim=1)
        score_n_pop_s = (user_emb_G_s_pop[user] * item_emb_G_s_pop[source_neg_item]).sum(dim=1)

        score_p_int_t = (user_emb_G_t_int[user] * item_emb_G_t_int[target_pos_item]).sum(dim=1)
        score_n_int_t = (user_emb_G_t_int[user] * item_emb_G_t_int[target_neg_item]).sum(dim=1)
        score_p_pop_t = (user_emb_G_t_pop[user] * item_emb_G_t_pop[target_pos_item]).sum(dim=1)
        score_n_pop_t = (user_emb_G_t_pop[user] * item_emb_G_t_pop[target_neg_item]).sum(dim=1)
 
        score_p_total_s = score_p_int_s + score_p_pop_s
        score_n_total_s = score_n_int_s + score_n_pop_s
        score_p_total_t = score_p_int_t + score_p_pop_t
        score_n_total_t = score_n_int_t + score_n_pop_t

        loss_int_s = self.mask_bpr_loss(score_p_int_s, score_n_int_s, mask_s)
        loss_pop_s = self.mask_bpr_loss(score_n_pop_s, score_p_pop_s, mask_s) + \
            self.mask_bpr_loss(score_p_pop_s, score_n_pop_s, ~mask_s)
        loss_int_t = self.mask_bpr_loss(score_p_int_t, score_n_int_t, mask_t)
        loss_pop_t = self.mask_bpr_loss(score_n_pop_t, score_p_pop_t, mask_t) + \
            self.mask_bpr_loss(score_p_pop_t, score_n_pop_t, ~mask_t)
        
        loss_total_s = self.bpr_loss(score_p_total_s, score_n_total_s)
        loss_total_t = self.bpr_loss(score_p_total_t, score_n_total_t)

        item_all_s = torch.unique(torch.cat((source_pos_item, source_neg_item)))
        item_all_t = torch.unique(torch.cat((target_pos_item, target_neg_item)))
        item_emb_int_s = self.source_item_embedding_int(item_all_s)
        item_emb_pop_s = self.source_item_embedding_pop(item_all_s)
        item_emb_int_t = self.target_item_embedding_int(item_all_t)
        item_emb_pop_t = self.target_item_embedding_pop(item_all_t)
        user_all = torch.unique(user)
        user_emb_int_s = self.source_user_embedding_int(user_all)
        user_emb_pop_s = self.source_user_embedding_pop(user_all)
        user_emb_int_t = self.target_user_embedding_int(user_all)
        user_emb_pop_t = self.target_user_embedding_pop(user_all)
        dis_loss = self.criterion_discrepancy(user_emb_int_s, user_emb_pop_s) + \
            self.criterion_discrepancy(item_emb_int_s, item_emb_pop_s) + self.criterion_discrepancy(user_emb_int_t, user_emb_pop_t) + \
            self.criterion_discrepancy(item_emb_int_t, item_emb_pop_t)
        
        loss = loss_total_s + loss_total_t + self.int_weight * (loss_int_s + loss_int_t) + self.pop_weight * (loss_pop_s + loss_pop_t) - self.dis_pen * dis_loss
        return loss
    
    def get_evaluate_embedding(self):
        if self.store_user_s == None:
           self.store_user_s, self.store_item_s, self.store_user_t, self.store_item_t = self.forward_emb('tot')

        source_user_feature_spe = self.store_user_s[self.user_index]
        target_user_feature_spe = self.store_user_t[self.user_index]
        source_item_feature_spe = self.store_item_s[self.source_item_index]
        target_item_feature_spe = self.store_item_t[self.target_item_index]
      
        return source_user_feature_spe, source_item_feature_spe, target_user_feature_spe, target_item_feature_spe


class LightGCN(nn.Module):
    def __init__(self, opt, s_adj, t_adj):
        super(LightGCN, self).__init__()
        self.opt=opt
        self.s_adj = s_adj
        self.t_adj = t_adj
        self.n_layers = opt['GNN']
        self.dropout = opt['dropout']
        self.emb_size = opt['feature_dim']
        self.temp = opt['temp']

        self.user_num = opt["source_user_num"]
        self.s_item_num = opt["source_item_num"]
        self.t_item_num = opt["target_item_num"]

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], self.emb_size)
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], self.emb_size)
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], self.emb_size)
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], self.emb_size)

        self.user_index = torch.arange(0, opt["source_user_num"], 1).cuda()
        self.source_user_index = torch.arange(0, opt["source_user_num"], 1).cuda()
        self.target_user_index = torch.arange(0, opt["target_user_num"], 1).cuda()
        self.source_item_index = torch.arange(0, opt["source_item_num"], 1).cuda()
        self.target_item_index = torch.arange(0, opt["target_item_num"], 1).cuda()

        # restore graph emb for val and test
        self.restore_user_s = None
        self.restore_item_s = None
        self.restore_user_t = None
        self.restore_item_t = None
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.source_user_embedding.weight.data)
        xavier_uniform_(self.target_user_embedding.weight.data)
        xavier_uniform_(self.source_item_embedding.weight.data)
        xavier_uniform_(self.target_item_embedding.weight.data)

    def predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        return output
    
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

        # return source_user, source_item, target_user, target_item
        all_embeddings_s = torch.cat([source_user, source_item], dim=0)  # (m+n1) * d
        all_embeddings_t = torch.cat([target_user, target_item], dim=0)  # (m+n2) * d

        user_G_emb_s, item_G_emb_s = self.lightGCN_forward(all_embeddings_s, self.s_adj, \
                                                           self.user_num, self.s_item_num)
        user_G_emb_t, item_G_emb_t = self.lightGCN_forward(all_embeddings_t, self.t_adj, \
                                                           self.user_num, self.t_item_num)
        return user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t
    
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
        
        user_G_emb_s, item_G_emb_s, user_G_emb_t, item_G_emb_t= self.get_G_emb()

        source_user_feature_spe = user_G_emb_s[user]
        target_user_feature_spe = user_G_emb_t[user]

        source_item_pos_feature_spe = item_G_emb_s[source_pos_item]
        target_item_pos_feature_spe = item_G_emb_t[target_pos_item]
        source_item_neg_feature_spe = item_G_emb_s[source_neg_item]
        target_item_neg_feature_spe = item_G_emb_t[target_neg_item]

        pos_source_score = self.predict_dot(source_user_feature_spe, source_item_pos_feature_spe)
        neg_source_score = self.predict_dot(source_user_feature_spe, source_item_neg_feature_spe)
        
        pos_target_score = self.predict_dot(target_user_feature_spe, target_item_pos_feature_spe)
        neg_target_score = self.predict_dot(target_user_feature_spe, target_item_neg_feature_spe)

        loss_bpr_source = torch.mean(torch.nn.functional.softplus(neg_source_score - pos_source_score)) 
        loss_bpr_target = torch.mean(torch.nn.functional.softplus(neg_target_score - pos_target_score))

        # get ego_emb for emb regular
        source_user_feature_ego = self.source_user_embedding(user)
        target_user_feature_ego = self.target_user_embedding(user)
        source_item_pos_feature_ego = self.source_item_embedding(source_pos_item)
        target_item_pos_feature_ego = self.target_item_embedding(target_pos_item)
        source_item_neg_feature_ego = self.source_item_embedding(source_neg_item)
        target_item_neg_feature_ego = self.target_item_embedding(target_neg_item)
        # regularization of GNN parameters
        reg_loss = (1 / 2) * (source_user_feature_ego.norm(2).pow(2) +
                              target_user_feature_ego.norm(2).pow(2) +
                              source_item_pos_feature_ego.norm(2).pow(2) + 
                               target_item_pos_feature_ego.norm(2).pow(2) +
                                source_item_neg_feature_ego.norm(2).pow(2) + 
                                target_item_neg_feature_ego.norm(2).pow(2)
                              ) / float(len(user))
        loss_rec = loss_bpr_source + loss_bpr_target + self.opt['reg_weight']*reg_loss

        loss = loss_rec
        return loss
    
    def get_evaluate_embedding(self):
        if self.restore_user_s is None or self.restore_item_s is None:
            self.restore_user_s, self.restore_item_s, self.restore_user_t, self.restore_item_t =  self.get_G_emb()

        source_user_feature_spe = self.restore_user_s[self.user_index]
        target_user_feature_spe = self.restore_user_t[self.user_index]
        source_item_feature_spe = self.restore_item_s[self.source_item_index]
        target_item_feature_spe = self.restore_item_t[self.target_item_index]

        return source_user_feature_spe, source_item_feature_spe, target_user_feature_spe, target_item_feature_spe

