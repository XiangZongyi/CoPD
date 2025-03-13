import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils

from model.CoPD import CoPD
from model.DIDA_CDR import DIDA_CDR
from model.DisenCDR import DisenCDR
from model.GA_DTCDR import GA_DTCDR
from model.SingleDomainBaseline import DGCF, DICE, LightGCN


class Trainer():
    def __init__(self, opt, source_adj, target_adj, cross_adj, source_UV, source_VU, target_UV, target_VU):
        if opt['model_name'] == 'CoPD':
            self.model = CoPD(opt, source_adj, target_adj, cross_adj).cuda()
        elif opt['model_name'] == 'DIDA_CDR':
            # lambda1=lambda2=1, GNN=2
            opt['GNN'] = 2
            opt['lambda1'] = opt['lambda2'] = 1.0
            self.model = DIDA_CDR(opt, source_UV, source_VU, target_UV, target_VU).cuda()
        elif opt['model_name'] == 'DisenCDR':
            opt['GNN'] = 2
            # self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
            opt['lr_decay'] = 0 
            self.model = DisenCDR(opt, source_UV, source_VU, target_UV, target_VU).cuda()
        elif opt['model_name'] == 'GA_DTCDR':
            self.model = GA_DTCDR(opt, source_UV, source_VU, target_UV, target_VU).cuda()
        elif opt['model_name'] == 'DGCF':
            self.model = DGCF(opt, source_UV, target_UV, source_adj, target_adj).cuda()
        elif opt['model_name'] == 'DICE':
            self.model = DICE(opt, source_adj, target_adj).cuda()
        elif opt['model_name'] == 'LightGCN':
            self.model = LightGCN(opt, source_adj, target_adj).cuda()
        else:
            raise ValueError("The parameter model_name is wrong.")
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'], l2=opt['lr_decay'])
        self.source_user, self.source_item, self.target_user, self.target_item = None, None, None, None
    
    def unpack_batch_predict(self, batch):
        inputs = [Variable(b.cuda()) for b in batch]
        user_index = inputs[0]
        item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        inputs = [Variable(b.cuda()) for b in batch]
        user = inputs[0]
        source_pos_item = inputs[1]
        source_neg_item = inputs[2]
        target_pos_item = inputs[3]
        target_neg_item = inputs[4]
        source_pop_item = inputs[5]
        target_pop_item = inputs[6]
        return user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, source_pop_item, target_pop_item

    def source_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.source_user, user_index)
        item_feature = self.my_index_select(self.source_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.target_user, user_index)
        item_feature = self.my_index_select(self.target_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def evaluate_embedding(self):
        self.source_user, self.source_item, self.target_user, self.target_item = self.model.get_evaluate_embedding()

    def reconstruct_graph(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
            source_pop_item, target_pop_item = self.unpack_batch(batch)

        loss = self.model(user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
                                                                source_pop_item, target_pop_item)
        loss.backward()
        self.optimizer.step()
        return loss.item()