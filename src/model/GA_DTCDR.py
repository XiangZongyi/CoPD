import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.autograd import Variable
import numpy as np
import math
from model.GCN import GCN


class GA_DTCDR(nn.Module):
    def __init__(self, opt, source_UV, source_VU, target_UV, target_VU):
        super(GA_DTCDR, self).__init__()
        opt['lambda1'] = 0.001
        self.opt=opt

        self.user_item_embedding_A = source_UV
        self.item_user_embedding_A = source_VU
        self.user_item_embedding_B = target_UV
        self.item_user_embedding_B = target_VU
        
        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        self.user_index = self.user_index.cuda()
        self.source_item_index = self.source_item_index.cuda()
        self.target_item_index = self.target_item_index.cuda()
        
        self.user_W1_A = self.init_variable(opt["source_user_num"], opt["feature_dim"])
        self.user_W1_B = self.init_variable(opt["target_user_num"], opt["feature_dim"])
        self.item_W1_A = self.init_variable(opt["source_item_num"], opt["feature_dim"])
        self.item_W1_B = self.init_variable(opt["target_item_num"], opt["feature_dim"])
        
        self.user_W_Attention_A_A = self.init_variable(opt["source_user_num"],  opt["feature_dim"]) # the weights of Domain A for Domain A
        # self.user_W_Attention_B_A = (1 - self.user_W_Attention_A_A)  # the weights of Domain B for Domain A
        self.user_W_Attention_B_B = self.init_variable(opt["target_user_num"],  opt["feature_dim"]) # the weights of Domain B for Domain B
        # self.user_W_Attention_A_B = (1 - self.user_W_Attention_B_B) # the weights of Domain A for Domain B
        
        K_size = opt["feature_dim"]
        self.userLayer = [ 
                          K_size, 2 * K_size, 4 * K_size, 8 * K_size,
                            4 * K_size, 2 * K_size, K_size
                        ]
        self.mlp_user_a = []
        self.mlp_item_a = []
        self.mlp_user_b = []
        self.mlp_item_b = []
        for i in range(0, len(self.userLayer) - 1):
            self.mlp_user_a.append(nn.Linear(self.userLayer[i], self.userLayer[i + 1]))
            self.mlp_user_a.append(nn.ReLU())
            self.mlp_item_a.append(nn.Linear(self.userLayer[i], self.userLayer[i + 1]))
            self.mlp_item_a.append(nn.ReLU())
            self.mlp_user_b.append(nn.Linear(self.userLayer[i], self.userLayer[i + 1]))
            self.mlp_user_b.append(nn.ReLU())
            self.mlp_item_b.append(nn.Linear(self.userLayer[i], self.userLayer[i + 1]))
            self.mlp_item_b.append(nn.ReLU())
        self.mlp_user_a = nn.ModuleList(self.mlp_user_a)
        self.mlp_item_a = nn.ModuleList(self.mlp_item_a)
        self.mlp_user_b = nn.ModuleList(self.mlp_user_b)
        self.mlp_item_b = nn.ModuleList(self.mlp_item_b)
        
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        
    def init_variable(self, dim1, dim2):
        v = nn.Parameter(torch.empty(dim1, dim2))
        nn.init.normal_(v, std=0.01)
        return v

    def predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def forward(self, user, source_pos_item, source_neg_item, target_pos_item, target_neg_item, \
            source_pop_item, target_pop_item):
        user_out_A = torch.spmm(self.user_item_embedding_A, self.item_W1_A[self.source_item_index])[user]
        item_out_A_pos = torch.spmm(self.item_user_embedding_A, self.user_W1_A[self.user_index])[source_pos_item]
        item_out_A_neg = torch.spmm(self.item_user_embedding_A, self.user_W1_A[self.user_index])[source_neg_item]
    
        user_out_B = torch.spmm(self.user_item_embedding_B, self.item_W1_B[self.target_item_index])[user]
        item_out_B_pos = torch.spmm(self.item_user_embedding_B, self.user_W1_B[self.user_index])[target_pos_item]
        item_out_B_neg = torch.spmm(self.item_user_embedding_B, self.user_W1_B[self.user_index])[target_neg_item]
        
        # Element-wise Attention for common users
        user_W_Attention_A_A_lookup = self.user_W_Attention_A_A[user]  # u*k
        # user_out_A_Combined = torch.add(
        #         torch.multiply(user_out_A, user_W_Attention_A_A_lookup),
        #         torch.multiply(user_out_B, user_W_Attention_B_A_lookup))
        user_out_A_Combined = torch.add(
            torch.multiply(user_out_A-user_out_B, user_W_Attention_A_A_lookup),
            user_out_B)
        
        user_W_Attention_B_B_lookup = self.user_W_Attention_B_B[user]
        # user_out_B_Combined = torch.add(
        #         torch.multiply(user_out_A, user_W_Attention_A_B_lookup),
        #         torch.multiply(user_out_B, user_W_Attention_B_B_lookup))
        user_out_B_Combined = torch.add(
            torch.multiply(user_out_A-user_out_B, user_W_Attention_B_B_lookup),
            user_out_B)
        user_out_A = user_out_A_Combined
        user_out_B = user_out_B_Combined
        
        # full-connected layers (MLP)
        for layer in self.mlp_user_a:
            user_out_A = layer(user_out_A)
        for layer in self.mlp_item_a:
            item_out_A_pos = layer(item_out_A_pos)
            item_out_A_neg = layer(item_out_A_neg)
        for layer in self.mlp_user_b:
            user_out_B = layer(user_out_B)
        for layer in self.mlp_item_b:
            item_out_B_pos = layer(item_out_B_pos)
            item_out_B_neg = layer(item_out_B_neg)
        
        pos_source_score = self.sigmoid(self.predict_dot(user_out_A, item_out_A_pos))
        neg_source_score = self.sigmoid(self.predict_dot(user_out_A, item_out_A_neg))
        
        pos_target_score = self.sigmoid(self.predict_dot(user_out_B, item_out_B_pos))
        neg_target_score = self.sigmoid(self.predict_dot(user_out_B, item_out_B_neg))
        
        regularizer_A = user_out_A.norm(2).pow(2) + item_out_A_pos.norm(2).pow(2) + item_out_A_neg.norm(2).pow(2)
        regularizer_B = user_out_B.norm(2).pow(2) + item_out_B_pos.norm(2).pow(2) + item_out_B_neg.norm(2).pow(2)
        
        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            pos_source_score.size())
        
        pos_labels = pos_labels.cuda()
        neg_labels = neg_labels.cuda()
        
        loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels)
        reg_loss = (1/2) * (regularizer_A + regularizer_B)
        return loss + self.opt['lambda1']*(reg_loss)
    
    def get_evaluate_embedding(self):
        user_out_A = torch.spmm(self.user_item_embedding_A, self.item_W1_A[self.source_item_index])
        item_out_A = torch.spmm(self.item_user_embedding_A, self.user_W1_A[self.user_index])
    
        user_out_B = torch.spmm(self.user_item_embedding_B, self.item_W1_B[self.target_item_index])
        item_out_B = torch.spmm(self.item_user_embedding_B, self.user_W1_B[self.user_index])
        
        # Element-wise Attention for common users
        user_W_Attention_A_A_lookup = self.user_W_Attention_A_A[self.user_index]  # u*k
        user_W_Attention_B_A_lookup = self.user_W_Attention_B_A[self.user_index]
        user_out_A_Combined = torch.add(
                torch.multiply(user_out_A, user_W_Attention_A_A_lookup),
                torch.multiply(user_out_B, user_W_Attention_B_A_lookup))
        
        user_W_Attention_B_B_lookup = self.user_W_Attention_B_B[self.user_index]
        user_W_Attention_A_B_lookup = self.user_W_Attention_A_B[self.user_index]
        user_out_B_Combined = torch.add(
                torch.multiply(user_out_A, user_W_Attention_A_B_lookup),
                torch.multiply(user_out_B, user_W_Attention_B_B_lookup))
        user_out_A = user_out_A_Combined
        user_out_B = user_out_B_Combined

        # full-connected layers (MLP)
        for layer in self.mlp_user_a:
            user_out_A = layer(user_out_A)
        for layer in self.mlp_item_a:
            item_out_A = layer(item_out_A)
        for layer in self.mlp_user_b:
            user_out_B = layer(user_out_B)
        for layer in self.mlp_item_b:
            item_out_B = layer(item_out_B)

        return user_out_A, item_out_A, user_out_B, item_out_B
