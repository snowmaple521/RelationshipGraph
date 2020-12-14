"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from model.fc import FCNet
from model.graph_att_layer import GraphSelfAttentionLayer


class GAttNet(nn.Module):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.2, label_bias=True, 
                 num_heads=16, pos_emb_dim=-1):
        super(GAttNet, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num = dir_num #2
        self.label_num = label_num#11
        self.in_feat_dim = in_feat_dim #2048
        self.out_feat_dim = out_feat_dim#1024
        self.dropout = nn.Dropout(dropout) #0.2
        # FCNet((main): Sequential(
        #     (0): Dropout(p=0.2, inplace=False)
        #     (1): Linear(in_features=2048, out_features=1024, bias=True)))
        self.self_weights = FCNet([in_feat_dim, out_feat_dim], '', dropout)
        # FCNet(
        #     (main): Sequential(
        #     (0): Linear(in_features=11, out_features=1, bias=False)))
        self.bias = FCNet([label_num, 1], '', 0, label_bias)
        self.nongt_dim = nongt_dim #20
        self.pos_emb_dim = pos_emb_dim #-1
        neighbor_net = []
        #两层图注意层
        for i in range(dir_num):
            g_att_layer = GraphSelfAttentionLayer(
                                pos_emb_dim=pos_emb_dim,
                                num_heads=num_heads,
                                feat_dim=out_feat_dim,
                                nongt_dim=nongt_dim)
            neighbor_net.append(g_att_layer)
        #图注意力网络
        self.neighbor_net = nn.ModuleList(neighbor_net)


    def forward(self, v_feat, adj_matrix, pos_emb=None):
        """
        Args:
            v_feat: [batch_size,num_rois, feat_dim]
            adj_matrix: [batch_size, num_rois, num_rois, num_labels]torch.Size([40, 36, 36, 15])
            pos_emb: [batch_size, num_rois, pos_emb_dim]

        Returns:
            output: [batch_size, num_rois, feat_dim]
        """
        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}")
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0")
        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        adj_matrix = adj_matrix.float()

        adj_matrix_list = [adj_matrix, adj_matrix.transpose(1, 2)] #2:

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.self_weights(v_feat)

        output = self_feat
        neighbor_emb = [0] * self.dir_num
        for d in range(self.dir_num):
            # [batch_size,num_rois, nongt_dim,label_num]
            input_adj_matrix = adj_matrix_list[d][:, :, :nongt_dim, :]
            condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1) #按行相加 【batch_size,num_rois, nongt_dim】

            # [batch_size,num_rois, nongt_dim] 与condensed_adj_matrix 维度相同
            v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)

            # [batch_size,num_rois, out_feat_dim] 第d层的邻居嵌入
            neighbor_emb[d] = self.neighbor_net[d].forward(
                        self_feat, condensed_adj_matrix, pos_emb,
                        v_biases_neighbors)

            # [batch_size,num_rois, out_feat_dim]
            output = output + neighbor_emb[d]
        output = self.dropout(output)
        output = nn.functional.relu(output)

        return output #torch.Size([40, 36, 1024])
