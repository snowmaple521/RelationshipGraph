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
import math
from torch.nn.utils.weight_norm import weight_norm


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, nongt_dim=20, pos_emb_dim=-1,
                 num_heads=16, dropout=[0.2, 0.5]):
        """ Attetion module with vectorized version
        自注意力的KVQ：其中K和Q是来自原始特征和其K的权重矩阵W_k的转换得来的——K=X*W_k,同理Q也是这样，
        而V可以通过转换，W_v得来，也可以是原始特征X一个嵌入即可。所以，K和Q需要一个网络来学习其权重矩阵，W_k,W_q

        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim] 用于隐式关系
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation 如果关系显式，设为-1
            nongt_dim: number of objects consider relations per image 对象的数量考虑每个图像的关系
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.fc_dim = num_heads #16
        self.feat_dim = feat_dim #1024
        self.dim = (feat_dim, feat_dim, feat_dim)
        self.dim_group = (int(self.dim[0] / num_heads),
                          int(self.dim[1] / num_heads),
                          int(self.dim[2] / num_heads))#64，64，64
        self.num_heads = num_heads #16
        self.pos_emb_dim = pos_emb_dim #隐式关系64，显式关系-1
        #如果是隐式关系执行 多一层pair_pos_fc1网络
        if self.pos_emb_dim > 0:
            self.pair_pos_fc1 = FCNet([pos_emb_dim, self.fc_dim], None, dropout[0]) #【64，16，None】

        self.query = FCNet([feat_dim, self.dim[0]], None, dropout[0]) #非线性全连接层作为查询【1024，1024】
        # self.query =
        # (query): FCNet(
        #     (main): Sequential(
        #       (0): Dropout(p=0.2, inplace=False)
        #       (1): Linear(in_features=1024, out_features=1024, bias=True)
        # )
        # )
        self.nongt_dim = nongt_dim #20

        self.key = FCNet([feat_dim, self.dim[1]], None, dropout[0]) ##非线性全连接层作为键【1024，1024】
        #  self.key =
        #     (key): FCNet(
        #         (main): Sequential(
        #         (0): Dropout(p=0.2, inplace=False)
        #     (1): Linear(in_features=1024, out_features=1024, bias=True) )
        #     )

        #weight_norm：pytorch自带的权重归一化[16*1024,1024,(1,1)]
        #卷积层：卷积计算
        self.linear_out_ = weight_norm(nn.Conv2d(in_channels=self.fc_dim * feat_dim,
                                      out_channels=self.dim[2],
                                      kernel_size=(1, 1),
                                      groups=self.fc_dim), dim=None)
        # self.linear_out_ =
        # (linear_out_): Conv2d(16384, 1024, kernel_size=(1, 1), stride=(1, 1), groups=16)
        #
        # )

    def forward(self, roi_feat, adj_matrix,position_embedding, label_biases_att):
        """
        Args:
            roi_feat: [batch_size, N, feat_dim] 区域特征
            adj_matrix: [batch_size, N, nongt_dim] 关系矩阵
            position_embedding: [num_rois, nongt_dim, pos_emb_dim] 位置嵌入
        Returns:
            output: [batch_size, num_rois, ovr_feat_dim, output_dim]
        """
        batch_size = roi_feat.size(0) #320
        num_rois = roi_feat.size(1) #36
        nongt_dim = self.nongt_dim if self.nongt_dim < num_rois else num_rois #20
        # [batch_size,nongt_dim, feat_dim]
        nongt_roi_feat = roi_feat[:, :nongt_dim, :] #取出前nongt_dim的数据 [320,20,1024]

        # [batch_size,num_rois, self.dim[0] = feat_dim]  roi_feat:[320,36,1024]=,q_data:[320,36,1024]
        q_data = self.query(roi_feat)  #用于指导注意力的查询向量数据 Q

        # [batch_size,num_rois, num_heads, feat_dim /num_heads]
        #q_data:[320,36,1024].view = torch.Size([320, 36, 16, 64])
        q_data_batch = q_data.view(batch_size, num_rois, self.num_heads,
                                   self.dim_group[0])

        # [batch_size,num_heads, num_rois, feat_dim /num_heads]
        #[320,16,36,1024]
        q_data_batch = torch.transpose(q_data_batch, 1, 2)

        # [batch_size,nongt_dim, self.dim[1] = feat_dim]
        #nongt_roi_feat:[320,20,1024]---linear(1024,1024) = [320,20,1024]
        #用于指导注意力数据的键向量
        k_data = self.key(nongt_roi_feat)

        # [batch_size,nongt_dim, num_heads, feat_dim /num_heads]
        #[320,20,1024].view  = [320,20,16,64]
        k_data_batch = k_data.view(batch_size, nongt_dim, self.num_heads,
                                   self.dim_group[1])

        # [batch_size,num_heads, nongt_dim, feat_dim /num_heads]
        ##[320,16,20,1024]
        k_data_batch = torch.transpose(k_data_batch, 1, 2)

        # [batch_size,nongt_dim, feat_dim]
        #注意力的值向量 就是原始36个区域取出20区域的图像特征值:[320,20,1024]
        v_data = nongt_roi_feat

        # [batch_size, num_heads, num_rois, nongt_dim]
        #获取注意力特征信息，查询与键相乘
        aff = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))

        # aff_scale, [batch_size, num_heads, num_rois, nongt_dim] 【320，16，36，20】
        #计算注意力值的公式
        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
        # aff_scale, [batch_size,num_rois,num_heads, nongt_dim] 【320，36，16，20】
        aff_scale = torch.transpose(aff_scale, 1, 2)
        #注意力的权重值 alphaij
        weighted_aff = aff_scale
        #如果是隐式关系，需要位置嵌入
        if position_embedding is not None and self.pos_emb_dim > 0:
            # Adding goemetric features
            #position_embedding: [320,36, 20, 64]
            position_embedding = position_embedding.float()
            # [batch_size,num_rois * nongt_dim, emb_dim]
            # 【320，36*20,64】
            position_embedding_reshape = position_embedding.view((batch_size, -1, self.pos_emb_dim))

            # position_feat_1, [batch_size,num_rois * nongt_dim, fc_dim] 【320，36*20，16】
            position_feat_1 = self.pair_pos_fc1(position_embedding_reshape) #[320,720,64]*[64,16]=[320，720，16]
            position_feat_1_relu = nn.functional.relu(position_feat_1) #非线性激活

            # aff_weight, [batch_size,num_rois, nongt_dim, fc_dim]【320，36，20，16】
            aff_weight = position_feat_1_relu.view((batch_size, -1, nongt_dim, self.fc_dim))

            # aff_weight, [batch_size,num_rois, fc_dim, nongt_dim] 【320，36，16,20】
            aff_weight = torch.transpose(aff_weight, 2, 3)

            thresh = torch.FloatTensor([1e-6]).cuda()
            # weighted_aff, [batch_size,num_rois, fc_dim, nongt_dim] 【320，36，16,20】
            threshold_aff = torch.max(aff_weight, thresh) #设置阈值

            weighted_aff += torch.log(threshold_aff)  #加上阈值的log值
        #如果存在关系矩阵
        if adj_matrix is not None:
            # weighted_aff_transposed, [batch_size,num_rois, nongt_dim, num_heads]【320，36，20，16】
            weighted_aff_transposed = torch.transpose(weighted_aff, 2, 3)
            zero_vec = -9e15*torch.ones_like(weighted_aff_transposed)
            #[320，36, 20].view = 【320，36, 20，1】
            adj_matrix = adj_matrix.view(adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2], 1)
            adj_matrix_expand = adj_matrix.expand((-1, -1, -1,weighted_aff_transposed.shape[-1]))
            #torch.where：第一个是判断条件，第二个是符合条件的设置值，第三个是不满足条件的设置值。间接设置最小阈值为0
            weighted_aff_masked = torch.where(adj_matrix_expand > 0,weighted_aff_transposed,zero_vec)

            weighted_aff_masked = weighted_aff_masked + label_biases_att.unsqueeze(3)
            weighted_aff = torch.transpose(weighted_aff_masked, 2, 3)

        # aff_softmax, [batch_size, num_rois, fc_dim, nongt_dim]
        #将张量缩放到0-1之间并且和为1
        aff_softmax = nn.functional.softmax(weighted_aff, 3)

        # aff_softmax_reshape, [batch_size, num_rois*fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view((batch_size, -1, nongt_dim))

        # output_t, [batch_size, num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)

        # output_t, [batch_size*num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view((-1, self.fc_dim * self.feat_dim, 1, 1))

        # linear_out, [batch_size*num_rois, dim[2], 1, 1]
        linear_out = self.linear_out_(output_t)
        output = linear_out.view((batch_size, num_rois, self.dim[2]))
        return output
