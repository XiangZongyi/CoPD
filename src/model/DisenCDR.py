import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from model.GCN import GCN

class DisenCDR(nn.Module):
    def __init__(self, opt, source_UV, source_VU, target_UV, target_VU):
        super(DisenCDR, self).__init__()
        self.opt=opt
        self.source_UV = source_UV
        self.source_VU = source_VU
        self.target_UV = target_UV
        self.target_VU = target_VU

        self.source_specific_GNN = singleVBGE(opt)
        self.source_share_GNN = singleVBGE(opt)

        self.target_specific_GNN = singleVBGE(opt)
        self.target_share_GNN = singleVBGE(opt)

        self.share_GNN = crossVBGE(opt)

        self.dropout = opt["dropout"]

        # self.user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(opt["target_user_num"], opt["feature_dim"])

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)


        self.user_index = self.user_index.cuda()
        self.source_user_index = self.source_user_index.cuda()
        self.target_user_index = self.target_user_index.cuda()
        self.source_item_index = self.source_item_index.cuda()
        self.target_item_index = self.target_item_index.cuda()
        
        self.criterion = nn.BCEWithLogitsLoss()

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss
    
    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
                                                                source_pop_item, target_pop_item):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_user_share = self.source_user_embedding_share(self.source_user_index)
        target_user_share = self.target_user_embedding_share(self.target_user_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, self.source_UV, self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, self.target_UV, self.target_VU)

        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(source_user, self.source_UV, self.source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(target_user, self.target_UV, self.target_VU)

        mean, sigma, = self.share_GNN(source_user_share, target_user_share, self.source_UV, self.source_VU, self.target_UV, self.target_VU)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_share_kld = self._kld_gauss(mean, sigma, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(mean, sigma, target_user_mean, target_user_sigma)

        self.kld_loss =  share_kld_loss + self.opt["beta"] * source_share_kld + self.opt[
            "beta"] * target_share_kld

        # source_learn_user = self.source_merge(torch.cat((user_share, source_learn_specific_user), dim = -1))
        # target_learn_user = self.target_merge(torch.cat((user_share, target_learn_specific_user), dim = -1))
        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user

        # return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item
        
        source_user_feature = self.my_index_select(source_learn_user, user)
        source_item_pos_feature = self.my_index_select(source_learn_specific_item, source_pos_item)
        source_item_neg_feature = self.my_index_select(source_learn_specific_item, source_neg_item)

        target_user_feature = self.my_index_select(target_learn_user, user)
        target_item_pos_feature = self.my_index_select(target_learn_specific_item, target_pos_item)
        target_item_neg_feature = self.my_index_select(target_learn_specific_item, target_neg_item)
        
        pos_source_score = self.predict_dot(source_user_feature, source_item_pos_feature)
        neg_source_score = self.predict_dot(source_user_feature, source_item_neg_feature)
        pos_target_score = self.predict_dot(target_user_feature, target_item_pos_feature)
        neg_target_score = self.predict_dot(target_user_feature, target_item_neg_feature)

        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            pos_source_score.size())

        pos_labels = pos_labels.cuda()
        neg_labels = neg_labels.cuda()
        
        loss = self.criterion(pos_source_score, pos_labels) + \
            self.criterion(neg_source_score, neg_labels) + \
            self.criterion(pos_target_score, pos_labels) + \
            self.criterion(neg_target_score, neg_labels) + \
            self.source_specific_GNN.encoder[-1].kld_loss + \
            self.target_specific_GNN.encoder[-1].kld_loss + self.kld_loss
        return loss

    def wramup(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          target_UV, target_VU)
        self.kld_loss = 0
        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item

    def get_evaluate_embedding(self):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_user_share = self.source_user_embedding_share(self.source_user_index)
        target_user_share = self.target_user_embedding_share(self.target_user_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, self.source_UV, self.source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, self.target_UV, self.target_VU)

        mean, sigma, = self.share_GNN(source_user_share, target_user_share, self.source_UV, self.source_VU, self.target_UV, self.target_VU)

        user_share, share_kld_loss = self.reparameters(mean, sigma)

        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user

        return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item

class singleVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(singleVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer_single(opt))
        self.encoder.append(LastLayer_single(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
        return mean, sigma

class DGCNLayer_single(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer_single, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)


class LastLayer_single(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer_single, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj,VU_adj)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)

        self.kld_loss = user_kld + item_kld

        return user, item


    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return user, kld_loss

    def forward_item(self, ufea, vfea, UV_adj,VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)

        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = torch.cat((Item_ho_mean, vfea), dim=1)
        Item_ho_mean = self.item_union_mean(Item_ho_mean)

        Item_ho_logstd = torch.cat((Item_ho_logstd, vfea), dim=1)
        Item_ho_logstd = self.item_union_logstd(Item_ho_logstd)

        item, kld_loss = self.reparameters(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = torch.cat((User_ho_mean, ufea), dim=1)
        User_ho_mean = self.user_union_mean(User_ho_mean)

        User_ho_logstd = torch.cat((User_ho_logstd, ufea), dim=1)
        User_ho_logstd = self.user_union_logstd(User_ho_logstd)

        # user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return User_ho_mean, User_ho_logstd
        # return user, kld_loss


class crossVBGE(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(crossVBGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer_cross(opt))
        self.encoder.append(LastLayer_cross(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        learn_user_source = source_ufea
        learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)

        mean, sigma, = self.encoder[-1](learn_user_source, learn_user_target, source_UV_adj,
                                                         source_VU_adj, target_UV_adj, target_VU_adj)
        return mean, sigma


class DGCNLayer_cross(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer_cross, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.source_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        self.source_rate = self.source_rate.cuda()

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho = self.gc3(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho = self.gc4(target_User_ho, target_UV_adj)

        source_User = torch.cat((source_User_ho , source_ufea), dim=1)
        source_User = self.source_user_union(source_User)
        target_User = torch.cat((target_User_ho, target_ufea), dim=1)
        target_User = self.target_user_union(target_User)

        return self.source_rate * F.relu(source_User) +  (1 - self.source_rate) * F.relu(target_User), self.source_rate * F.relu(source_User) + (1 - self.source_rate) * F.relu(target_User)


class LastLayer_cross(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(LastLayer_cross, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],  # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leaky"]
        )
        self.source_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.source_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.target_user_union_logstd = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

        self.source_rate = torch.tensor(self.opt["rate"]).view(-1)

        self.source_rate = self.source_rate.cuda()


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        source_User_mean = torch.cat(
            (source_User_ho_mean, source_ufea), dim=1)
        source_User_mean = self.source_user_union_mean(source_User_mean)

        source_User_logstd = torch.cat((source_User_ho_logstd, source_ufea), dim=1)
        source_User_logstd = self.source_user_union_logstd(source_User_logstd)

        target_User_mean = torch.cat(
            (target_User_ho_mean, target_ufea), dim=1)
        target_User_mean = self.target_user_union_mean(target_User_mean)

        target_User_logstd = torch.cat(
            (target_User_ho_logstd, target_ufea),
            dim=1)
        target_User_logstd = self.target_user_union_logstd(target_User_logstd)

        return self.source_rate * source_User_mean + (1 - self.source_rate) * target_User_mean, self.source_rate * source_User_logstd + (1 - self.source_rate) * target_User_logstd
