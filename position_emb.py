"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import numpy as np
import math
import torch
from torch.autograd import Variable


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # 确定矩形交点的(x, y)坐标
    # 主要是衡量模型生成的boxA和boxB之间的重叠程度
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle 计算矩形交点的面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles 矩型
    # 计算预测和ground-truth的面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # 通过求交集来计算并集的交集
    # area and dividing it by the sum of prediction + ground-truth
    # 面积，然后除以预测值+地面真实值的和
    # areas - the interesection area 区域-兴趣区域
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    # 返回并集值的交集
    #IoU(Intersection over Union)，即交并比，是目标检测中常见的评价标准，主要是衡量模型生成的bounding box和ground truth box之间的重叠程度，
    return iou


def build_graph(bbox, spatial, label_num=11):
    """ Build spatial graph
    #构建空间图

    Args:
        bbox: [num_boxes, 4]
        实验传来：bbox：320，36，4
        实验传来：spatial：320，36，36

    Returns:
        adj_matrix: [num_boxes, num_boxes, label_num]
    """

    num_box = bbox.shape[1]
    batchsize = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    # [num_boxes, 1]
    # bbox_width:,bbox_height,xmin,[320,9,4]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    bbox_height = bbox_height.view(batchsize,num_box)
    bbox_width = bbox_width.view(batchsize,num_box)
    image_h = bbox_height[0]/spatial[0, -1]
    image_w = bbox_width[0]/spatial[0, -2]
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(image_h**2 + image_w**2)
    for i in range(num_box):
        bbA = bbox[i]
        if sum(bbA) == 0:
            continue
        adj_matrix[i, i] = 12
        for j in range(i+1, num_box):
            bbB = bbox[j]
            if sum(bbB) == 0:
                continue
            # class 1: inside (j inside i) #对象j在对象i中
            if xmin[i] < xmin[j] and xmax[i] > xmax[j] and \
               ymin[i] < ymin[j] and ymax[i] > ymax[j]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 2
            # class 2: cover (j covers i) #对象j包含对象i
            elif (xmin[j] < xmin[i] and xmax[j] > xmax[i] and
                  ymin[j] < ymin[i] and ymax[j] > ymax[i]):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)
                # class 3: i and j overlap i和j重叠
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff)**2 + (x_diff)**2)
                    if diag < 0.5 * image_diag:
                        sin_ij = y_diff/diag
                        cos_ij = x_diff/diag
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = 2*math.pi - label_i
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)+2*math.pi
                            label_j = label_i - math.pi
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = 2*math.pi - label_i
                        else:
                            label_i = -np.arccos(sin_ij)+2*math.pi
                            label_j = label_i - math.pi
                        adj_matrix[i, j] = int(np.ceil(label_i/(math.pi/4)))+3
                        adj_matrix[j, i] = int(np.ceil(label_j/(math.pi/4)))+3
    return adj_matrix


def torch_broadcast_adj_matrix(adj_matrix, label_num, device=torch.device("cuda")):
    """ broadcast spatial relation graph
    #广播空间关系图

    Args:
        #【1，36，36】
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:#【1，36，36，11】
        result: [batch_size,num_boxes, num_boxes, label_num]
    """
    result = []
    for i in range(1, label_num+1):
        index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze()
        curr_result = torch.zeros( adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2])
        curr_result = curr_result.view(-1)
        curr_result[index] += 1
        result.append(curr_result.view((adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2], 1)))
    result = torch.cat(result, dim=3) #【320，36，36，11】
    return result


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    '''
    :param position_mat: 位置矩阵
    :param feat_dim: 特征嵌入维度
    :param wave_length: 波长
    :param device:
    :return: 嵌入向量
    '''
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    # 将位置矩阵嵌入
    feat_range = torch.arange(0, feat_dim / 8) #torch.float32，feat_dim / 8个tensor数据
    dim_mat = torch.pow(torch.ones((1,))*wave_length,
                        (8. / feat_dim) * feat_range) #torch.Size([1, 1, 1, 8]) torch.float32 cuda
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device) #torch.Size([1, 1, 1, 8])
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4) #torch.Size([1, 20, 36, 4, 1]) torch.float32
    div_mat = torch.div(position_mat.to(device), dim_mat) #torch.Size([1, 20, 36, 4, 8]) torch.float32
    sin_mat = torch.sin(div_mat) #torch.Size([1, 20, 36, 4, 8]) torch.float32
    cos_mat = torch.cos(div_mat) #torch.Size([1, 20, 36, 4, 8]) torch.float32
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1) #torch.Size([1, 20, 36, 4, 16]) torch.float32
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], feat_dim) #torch.Size([320, 20, 36, 64]) torch.float32
    return embedding


def torch_extract_position_matrix(bbox, nongt_dim=36):
    # 提取位置矩阵
    """ Extract position matrix

    Args:
        #【1，36，4】
        bbox: [batch_size, num_boxes, 4]

    Returns:
        【1，36，20，4】
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1) #取出区域框中最大最小值，主要用于计算框的宽度和高度和中心位置
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1. #torch.Size([1, 36, 1]) torch.float32 cuda 框的宽度
    bbox_height = ymax - ymin + 1.#torch.Size([1, 36, 1]) torch.float32 cuda 框的高度
    center_x = 0.5 * (xmin + xmax) #torch.Size([1, 36, 1]) torch.float32 cuda 框的中心位置x
    center_y = 0.5 * (ymin + ymax) #torch.Size([1, 36, 1]) torch.float32 cuda 框的中心位置y
    # [batch_size,num_boxes, num_boxes]
    #torch.transpose(center_x, 1, 2) = [1,1,36]
    delta_x = center_x-torch.transpose(center_x, 1, 2) #torch.Size([1, 36, 36]) torch.float32  cuda
    #delta_x 再除以区域宽度
    delta_x = torch.div(delta_x, bbox_width) #torch.Size([1, 36, 36]) torch.float32  cuda
    #求其绝对值
    delta_x = torch.abs(delta_x)
    threshold = 1e-3 #设置最阈值
    #如果小于thredhold，就赋值threshold
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x) #取对数 以自然数为e取对数

    #同理求出delta_y
    delta_y = center_y-torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)

    #求出delta_宽度和高度
    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)
    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)


    #将这些数组合并，
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim] #取出20个对象即可 sym =【1，20，36】
        concat_list[idx] = torch.unsqueeze(sym, dim=3) #列表第i个元素增加一个维度，【1，20，36，1】
    position_matrix = torch.cat(concat_list, 3) #位置矩阵
    return position_matrix


def prepare_graph_variables(relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects, nongt_dim, pos_emb_dim, spa_label_num, sem_label_num, device):
    '''
    relation_type:关系类型
    bb:区域盒子位置
    sem_adj_matrix:语义邻居矩阵
    spa_adj_matrix: 空间邻居矩阵
    num_objects:对象数 36
    nongt_dim:经过考虑每个图像的关系，所选对象数量 20
    pos_emb_dim:位置嵌入维度64
    spa_label_num:空间标签数目11
    sem_label_num: 语义标签数目15
    device:设备gpu
    '''

    pos_emb_var, sem_adj_matrix_var, spa_adj_matrix_var = None, None, None
    if relation_type == "spatial":
        assert spa_adj_matrix.dim() > 2, "Found spa_adj_matrix of wrong shape"
        spa_adj_matrix = spa_adj_matrix.to(device)
        #tensor分割，取出=区域数的维度
        ##如果空间大于36维度，只取空间是前36的
        spa_adj_matrix = spa_adj_matrix[:, :num_objects, :num_objects]
        #3维变4维，多一维空间关系：共11个空间关系决策
        spa_adj_matrix = torch_broadcast_adj_matrix( spa_adj_matrix, label_num=spa_label_num, device=device)
        spa_adj_matrix_var = Variable(spa_adj_matrix).to(device)
    if relation_type == "semantic":
        assert sem_adj_matrix.dim() > 2, "Found sem_adj_matrix of wrong shape"
        sem_adj_matrix = sem_adj_matrix.to(device)
        sem_adj_matrix = sem_adj_matrix[:, :num_objects, :num_objects]
        # sem_adj_matrix = build_graph(bb,sem_adj_matrix,label_num=sem_label_num)
        sem_adj_matrix = torch_broadcast_adj_matrix(sem_adj_matrix, label_num=sem_label_num, device=device)
        ##3维变4维，多一维空间关系：共15个语义关系备选
        sem_adj_matrix_var = Variable(sem_adj_matrix).to(device)
    else:#如果是语义关系，就没有位置矩阵，如果是空间或者隐式关系就有位置矩阵。
        bb = bb.to(device) #torch.float32 cuda:0 #【320，36，4】
        pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim) #位置矩阵 torch.Size([320, 20, 36, 4])
        pos_emb = torch_extract_position_embedding( pos_mat, feat_dim=pos_emb_dim, device=device) # torch.Size([1, 20, 36, 64]) torch.float32 cuda
        pos_emb_var = Variable(pos_emb).to(device) # torch.Size([1, 20, 36, 64]) torch.float32 cuda
    return pos_emb_var, sem_adj_matrix_var, spa_adj_matrix_var
