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
from model.fusion import BAN, BUTD, MuTAN
from model.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from model.classifier import SimpleClassifier


class ReGAT(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_embedding, classifier, glimpse, fusion, relation_type):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.classifier = classifier

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]torch.Size([1, 14]) torch.int32
        labels:torch.Size([1, 3129]) torch.int64
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels] 【1024，36，15】
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels] 【1024，36，11】

        return: logits, not probs
        """
#        q = q.type(torch.LongTensor).cuda()
        w_emb = self.w_emb(q) #torch.Size([40, 14, 600]) torch.float32 cuda 40=320/8
        q_emb_seq = self.q_emb.forward_all(w_emb)  #torch.Size([1, 14, 1024]) torch.float32 cuda [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq) #torch.Size([1, 1024]) torch.float32 cuda

        # [batch_size, num_rois, out_dim]
        if self.relation_type == "semantic": #如果是语义，传递语义关系。
            v_emb = self.v_relation.forward(v, sem_adj_matrix, q_emb_self_att)
        elif self.relation_type == "spatial": #如果是空间，传递空间关系
            v_emb = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # implicit
            v_emb = self.v_relation.forward(v, implicit_pos_emb,
                                            q_emb_self_att) #torch.Size([1, 36, 1024]) torch.float32 cuda

        if self.fusion == "ban":
            joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
        elif self.fusion == "butd":
            q_emb = self.q_emb(w_emb)  # [batch, q_dim]
            joint_emb, att = self.joint_embedding(v_emb, q_emb)
        else:  # mutan
            #joint_emb:torch.Size([1, 3129]) torch.float32 cuda
            #att:torch.Size([1, 2048]) torch.float32 cuda
            joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att) #
        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb
        return logits, att


def build_regat(dataset, args):
    print("Building ReGAT model with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    # 词嵌入向量模型
    # WordEmbedding(
    #   (emb): Embedding(19902, 300, padding_idx=19901)
    #   (emb_): Embedding(19902, 300, padding_idx=19901)
    #   (dropout): Dropout(p=0.0, inplace=False)
    # )
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op) #调用WordEmbding词嵌入方法
    #问题嵌入
    #  QuestionEmbedding(
    # (rnn): GRU(600, 1024, batch_first=True) )
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,args.num_hid, 1, False, .0)
    #问题自注意力
        # QuestionSelfAttention(
    #   (drop): Dropout(p=0.2, inplace=False)
    #   (W1_self_att_q): FCNet(
    #     (main): Sequential(
    #       (0): Dropout(p=0.2, inplace=False)
    #       (1): Linear(in_features=1024, out_features=1024, bias=True)
    #     )
    #   )
    #   (W2_self_att_q): FCNet(
    #     (main): Sequential(
    #       (0): Linear(in_features=1024, out_features=1, bias=True)
    #     )
    #   )
    # )
    q_att = QuestionSelfAttention(args.num_hid, .2)

    if args.relation_type == "semantic": #如果关系类型是语义的
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "spatial": #如果关系类型是空间的
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    else: #否则是隐式关系
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads=args.num_heads, num_steps=args.num_steps,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    #分类器
    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    gamma = 0
    #采用融合方法
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    return ReGAT(dataset, w_emb, q_emb, q_att, v_relation, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type)
