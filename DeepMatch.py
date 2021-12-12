#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from o3d_registration import fgr_registration


# 计算获得最远点的下标和距离
def get_furthest(x):
    # batch_size * 3 * num_points
    inner = 2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx + xx.transpose(2, 1).contiguous() - inner

    dist, idx = pairwise_distance.max(dim=-1)  # batch_size * num_points
    return dist, idx


# 得到点云的逐点结构
def get_graph_feature(x):
    batch_size, _, num_points = x.size()
    x = x - x.mean(dim=-1, keepdim=True)
    xx = torch.sum(x ** 2, dim=1, keepdim=False)
    x *= 1.0 / (torch.max(xx) ** 0.5)

    dist, idx = get_furthest(x)
    dist = dist.view(batch_size, num_points, 1, 1)

    idx_base = torch.arange(0, batch_size).view(-1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    furthest = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k) * 3
    xx = torch.sum(x ** 2, dim=-1, keepdim=True).view(batch_size, num_points, 1, -1)
    furthest_dist = torch.sum(furthest ** 2, dim=-1, keepdim=True).view(batch_size, num_points, 1, -1)
    feature = torch.cat((xx, furthest_dist, dist), dim=-1)  # batch_size * num_ points * 1 * 3

    return feature.permute(0, 3, 1, 2)


# 主干网络
class backBone(nn.Module):
    def __init__(self, emb_dims=128):
        super(backBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        return x.view(batch_size, -1, num_points)


# 使用 SVD 通过特征 embedding 获得刚体变换
class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())

            R.append(r)

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


class DeepMatch(nn.Module):
    def __init__(self):
        super(DeepMatch, self).__init__()
        self.emb_nn = backBone(emb_dims=128)
        self.head = SVDHead()

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        rotation, translation = self.head(src_embedding, tgt_embedding, src, tgt)

        return rotation, translation


# 获得上下偏差并显示
def get_error(pc_tgt, pc_res, mn, mx):
    # 计算点对之间的距离dist以及它们到中心点的距离之差error
    # mn < error < mx（阈值），则认为没有偏差或不是对应点
    # error小于0为上偏差，大于0为下偏差
    # point_nums * 3
    tgt = torch.from_numpy(np.asarray(pc_tgt.points)).float()
    res = torch.from_numpy(np.asarray(pc_res.points)).float()
    tgt_centered = tgt.mean(dim=0, keepdim=True)

    # 计算点到点距离
    inner = 2 * torch.matmul(tgt, res.transpose(1, 0).contiguous())
    tgt_inner = torch.sum(tgt ** 2, dim=1, keepdim=True)
    res_inner = torch.sum(res ** 2, dim=1, keepdim=True).transpose(1, 0)
    pairwise_dist = tgt_inner + res_inner - inner
    pairwise_dist = pairwise_dist ** 0.5

    # 获取最近点距离
    dist, idx = torch.min(pairwise_dist, dim=1)
    pairwise = res[idx, :]

    # tgt 到中心点的距离
    tgt_centered_inner = torch.sum(tgt_centered ** 2, dim=1, keepdim=True)
    inner = 2 * torch.matmul(tgt, tgt_centered.transpose(1, 0).contiguous())
    tgt_2_center_dist = (tgt_centered_inner + tgt_inner - inner) ** 0.5

    # pairwise 到中心点的距离
    pairwise_inner = torch.sum(pairwise ** 2, dim=1, keepdim=True)
    inner = 2 * torch.matmul(pairwise, tgt_centered.transpose(1, 0).contiguous())
    pairwise_2_center_dist = (tgt_centered_inner + pairwise_inner - inner) ** 0.5

    # 计算每个点的偏差
    pairwise_error = tgt_2_center_dist - pairwise_2_center_dist

    # 将原点， 最近点， 距离拼成一个矩阵
    pair = torch.cat((tgt, pairwise, pairwise_error), dim=1)

    np.savetxt("data/error.txt", pair.numpy(), fmt='%f', delimiter='   ')

    # 返回上偏差点云和下偏差点云
    pc_upper = o3d.geometry.PointCloud()
    pc_down = o3d.geometry.PointCloud()
    upper_idx = torch.where((pairwise_error < -mn) & (pairwise_error > -mx))[0]
    down_idx = torch.where((pairwise_error > mn) & (pairwise_error < mx))[0]
    pc_upper.points = o3d.utility.Vector3dVector(pairwise[upper_idx, :].numpy())
    pc_down.points = o3d.utility.Vector3dVector(pairwise[down_idx, :].numpy())
    return pc_upper, pc_down


def deepmatch_registration(pc_src, pc_tgt, voxel_size=50.0):
    # voxel = 0.5 means 50cm for this dataset 采样参数
    # get model
    # init
    net = DeepMatch()
    net.load_state_dict(torch.load("pretrained/model.best.t7", map_location='cpu'))

    pc_src_down = o3d.geometry.PointCloud.uniform_down_sample(pc_src, len(pc_src.points) // 1024)
    pc_tgt_down = o3d.geometry.PointCloud.uniform_down_sample(pc_tgt, len(pc_tgt.points) // 1024)

    src = torch.from_numpy(np.asarray(pc_src_down.points)[:1024][:]).transpose(1, 0).contiguous().view(-1, 3,
                                                                                                       1024).float()
    tgt = torch.from_numpy(np.asarray(pc_tgt_down.points)[:1024][:]).transpose(1, 0).contiguous().view(-1, 3,
                                                                                                       1024).float()

    # DeepMatch part
    net.eval()

    rotation_pred, translation_pred = net(src, tgt)
    transformation = torch.cat((rotation_pred.view(3, 3), translation_pred.transpose(0, 1).contiguous()), dim=1)
    transformation = torch.cat((transformation, torch.tensor([[0, 0, 0, 1]])), dim=0)
    transformation = transformation.detach().numpy()
    # done DeepMatch part

    pc_src_down.transform(transformation)
    transformation = np.dot(fgr_registration(pc_src_down, pc_tgt_down), transformation)

    # icp part
    radius_normal = voxel_size * 2
    icp_distance_threshold = voxel_size * 0.5

    pc_src_icp = o3d.geometry.PointCloud()
    pc_tgt_icp = o3d.geometry.PointCloud()
    if len(pc_src.points) > 5000 and len(pc_tgt.points) > 5000:
        pc_src_icp = o3d.geometry.PointCloud.uniform_down_sample(pc_src, len(pc_src.points) // 5000)
        pc_tgt_icp = o3d.geometry.PointCloud.uniform_down_sample(pc_tgt, len(pc_tgt.points) // 5000)

    # estimate normals
    pc_src_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pc_tgt_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    pc_res_icp = o3d.pipelines.registration.registration_icp(pc_src_icp, pc_tgt_icp, icp_distance_threshold,
                                                             transformation,
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return pc_res_icp.transformation
    # done icp part

    # match
    # pc_res = copy.deepcopy(pc_src)
    # pc_res.transform(pc_res_icp.transformation)
    #
    # pc_tgt_down = o3d.geometry.PointCloud.uniform_down_sample(pc_tgt, len(pc_tgt.points) // 10000)
    # pc_res_down = o3d.geometry.PointCloud.uniform_down_sample(pc_res, len(pc_res.points) // 10000)
    #
    # pc_error_upper, pc_error_down = get_error(pc_tgt_down, pc_res_down, 0.1, 1)
    #
    # pc_res.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pc_res, pc_tgt])
    #
    # pc_tgt.paint_uniform_color([0.4, 0.4, 0.4])
    # pc_error_upper.paint_uniform_color([1, 0, 0])
    # pc_error_down.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pc_error_down, pc_error_upper, pc_tgt])


if __name__ == "__main__":
    pass