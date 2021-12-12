#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import torch
import numpy as np
import open3d as o3d


# 点云拼接，返回刚体变换
def deepmerge(pc_src, pc_tgt, voxel_size=50.0):
    # voxel = 0.5 means 50cm for this dataset 采样参数
    # estimate normals
    pc_src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pc_tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # ICP part
    # estimate normals
    pc_src_icp = copy.deepcopy(pc_src)
    pc_tgt_icp = copy.deepcopy(pc_tgt)

    # ICP 算法的初始刚体变换
    pc_res_icp = o3d.pipelines.registration.registration_icp(pc_src_icp, pc_tgt_icp, voxel_size * 0.5, np.identity(4),
                                                             o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # align
    return pc_res_icp.transformation


# 获得待拼接两片点云形状近似相同的部分
def get_same_part(pc_src, pc_tgt, GET_SAME_MAX_DIST=0.05):
    # 利用 torch 计算得到两片点云形状近似相同的部分
    # 将数据转换成张量
    tgt = torch.from_numpy(np.asarray(pc_tgt.points)).float()
    src = torch.from_numpy(np.asarray(pc_src.points)).float()

    # 计算源点云中所有点到目标点云中所有点的距离
    inner = 2 * torch.matmul(src, tgt.transpose(1, 0).contiguous())
    src_inner = torch.sum(src ** 2, dim=1, keepdim=True)
    tgt_inner = torch.sum(tgt ** 2, dim=1, keepdim=True).transpose(1, 0)
    pairwise_dist = src_inner + tgt_inner - inner
    pairwise_dist = pairwise_dist ** 0.5

    # 获得源点云中形状近似相同的部分
    src_to_tgt_dist, _ = torch.min(pairwise_dist, dim=1)  # 获得源点云中所有点到其在目标点云中最近点的距离
    # 若源点云中的某个点其在目标点云中最近点的距离小于 GET_SAME_MAX_DIST，则认为源点云中的这个点属于形状近似相同部分
    src_to_tgt_closet_idx = torch.where(src_to_tgt_dist < GET_SAME_MAX_DIST)
    src_same = src[src_to_tgt_closet_idx]

    # 获得目标点云中形状近似相同的部分
    tgt_to_src_dist, _ = torch.min(pairwise_dist, dim=0)  # 获得目标点云中所有点到其在源点云中最近点的距离
    # 若目标点云中的某个点其在源点云中最近点的距离小于 GET_SAME_MAX_DIST，则认为目标点云中的这个点属于形状近似相同部分
    tgt_to_src_closet_idx = torch.where(tgt_to_src_dist < GET_SAME_MAX_DIST)
    tgt_same = tgt[tgt_to_src_closet_idx]

    src_same = src_same.detach().numpy()
    tgt_same = tgt_same.detach().numpy()

    # 定义两个点云，保存源点云和目标点云中形状近似相同的部分
    pc_src_closet = o3d.geometry.PointCloud()
    pc_tgt_closet = o3d.geometry.PointCloud()
    pc_src_closet.points = o3d.utility.Vector3dVector(src_same)
    pc_tgt_closet.points = o3d.utility.Vector3dVector(tgt_same)
    return pc_src_closet, pc_tgt_closet


# 在源点云经过刚体变换完成拼接后，去除源点云中形状近似相同的部分
def del_same_in_src(pc_src, pc_tgt, DEL_SAME_MAX_DIST=3.0):
    # 利用 torch 计算得到两片点云形状近似相同的部分，并删除源点云中形状近似相同部分的点
    # 将数据转换成张量
    tgt = torch.from_numpy(np.asarray(pc_tgt.points)).float()
    src = torch.from_numpy(np.asarray(pc_src.points)).float()

    # 计算源点云中所有点到目标点云中所有点的距离
    inner = 2 * torch.matmul(src, tgt.transpose(1, 0).contiguous())
    src_inner = torch.sum(src ** 2, dim=1, keepdim=True)
    tgt_inner = torch.sum(tgt ** 2, dim=1, keepdim=True).transpose(1, 0)
    pairwise_dist = src_inner + tgt_inner - inner
    pairwise_dist = pairwise_dist ** 0.5

    # 获得源点云中形状近似相同部分之外的点云
    src_to_tgt_dist, _ = torch.min(pairwise_dist, dim=1)
    # 若目标点云中的某个点其在源点云中最近点的距离大于 DEL_SAME_MAX_DIST，则认为目标点云中的这个点不属于形状近似相同部分
    src_to_tgt_closet_idx = torch.where(src_to_tgt_dist > DEL_SAME_MAX_DIST)
    src_del_same = src[src_to_tgt_closet_idx]

    # 保存并返回去除形状相同部分之后的源点云
    src_del_same = src_del_same.detach().numpy()

    pc_src_del_same = o3d.geometry.PointCloud()
    pc_src_del_same.points = o3d.utility.Vector3dVector(src_del_same)

    return pc_src_del_same


# 两片点云完成拼接并显示
def merge(pc_src, pc_tgt, GET_SAME_MAX_DIST=15.0):
    pc_src_down = o3d.geometry.PointCloud.uniform_down_sample(pc_src, len(pc_src.points) // 10000)
    pc_tgt_down = o3d.geometry.PointCloud.uniform_down_sample(pc_tgt, len(pc_tgt.points) // 10000)

    pc_src_closest, pc_tgt_closest = get_same_part(pc_src_down, pc_tgt_down, GET_SAME_MAX_DIST)
    if len(pc_src_closest.points) < 500:
        return False, None
    transformation = deepmerge(pc_src_closest, pc_tgt_closest)

    return True, transformation


if __name__ == "__main__":
    pass
