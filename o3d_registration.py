#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import open3d as o3d


# fpfh 粗配准加 ICP 精配准
def fpfh_icp(src_path, tgt_path, voxel_size=50.0):
    # voxel = 0.5 means 50cm for this dataset 采样参数
    pc_src = o3d.io.read_point_cloud(src_path)
    pc_tgt = o3d.io.read_point_cloud(tgt_path)

    # params
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    fpfh_distance_threshold = voxel_size * 1.5
    icp_distance_threshold = voxel_size * 0.5

    # fpfh part
    # voxel sampling
    pc_src_down = pc_src.voxel_down_sample(voxel_size)
    pc_tgt_down = pc_tgt.voxel_down_sample(voxel_size)

    # estimate normals
    pc_src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pc_tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # compute fpfh
    pc_src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pc_src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pc_tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pc_tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    pc_res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pc_src_down, pc_tgt_down, pc_src_fpfh, pc_tgt_fpfh, True, fpfh_distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                fpfh_distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    # done fpfh part

    # icp part
    # 估计过法向量的点云会显示法向量，所以拷贝一份用于估计法向量
    pc_src_icp = copy.deepcopy(pc_src)
    pc_tgt_icp = copy.deepcopy(pc_tgt)

    # estimate normals
    pc_src_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pc_tgt_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    pc_res_icp = o3d.pipelines.registration.registration_icp(pc_src_icp, pc_tgt_icp, icp_distance_threshold,
                                                             pc_res_ransac.transformation,
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # align
    return pc_res_icp.transformation


def icp_registration(pc_src, pc_tgt, voxel_size=50.0):
    # voxel = 0.5 means 50cm for this dataset 采样参数
    pc_src_icp = o3d.geometry.PointCloud.uniform_down_sample(pc_src, len(pc_src.points) // 5000)
    pc_tgt_icp = o3d.geometry.PointCloud.uniform_down_sample(pc_tgt, len(pc_tgt.points) // 5000)

    # estimate normals
    pc_src_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pc_tgt_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pc_res_icp = o3d.pipelines.registration.registration_icp(pc_src_icp, pc_tgt_icp, 2 * voxel_size, np.identity(4),
                                                             o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return pc_res_icp.transformation


def fgr_registration(pc_src, pc_tgt, voxel_size=50.0):
    # voxel = 0.5 means 50cm for this dataset 采样参数
    pc_src_fgr = copy.deepcopy(pc_src)
    pc_tgt_fgr = copy.deepcopy(pc_tgt)

    # estimate normals
    pc_src_fgr.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pc_tgt_fgr.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    pc_src_fgr_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_src_fgr,
                                                                      o3d.geometry.KDTreeSearchParamHybrid(
                                                                          radius=voxel_size * 2, max_nn=100))
    pc_tgt_fgr_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_tgt_fgr,
                                                                      o3d.geometry.KDTreeSearchParamHybrid(
                                                                          radius=voxel_size * 2, max_nn=100))

    pc_res_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc_src, pc_tgt, pc_src_fgr_fpfh, pc_tgt_fgr_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 2))

    return pc_res_fgr.transformation


if __name__ == "__main__":
    pass
