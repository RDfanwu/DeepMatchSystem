#!/user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


# 得到一个随机刚体变换，该刚体变换以每个坐标轴为旋转轴生成一个[0, angle]的旋转，沿每个坐标轴方向生成一个[-distance, distance]的平移
def get_rd_transformation(angle=6, distance=0.01):
    anglex = np.random.uniform() * np.pi * angle / 360
    angley = np.random.uniform() * np.pi * angle / 360
    anglez = np.random.uniform() * np.pi * angle / 360

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    t = np.array([[np.random.uniform(-distance, distance)], [np.random.uniform(-distance, distance)],
                  [np.random.uniform(-distance, distance)]])
    transformation = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
    return transformation


# 不改变点云的中心点坐标进行点云放缩，放缩的倍数为 multiple
def scaling(path, multiple=10):
    pc = o3d.io.read_point_cloud(path)
    pc.scale(multiple, center=pc.get_center())
    o3d.io.write_point_cloud(path, pc, write_ascii=True)


# 给点云加上高斯噪声
def add_gn(pc):
    src = np.asarray(pc.points)
    N, C = src.shape
    src = src + np.clip(0.01 * np.random.randn(N, C), -1 * 0.05, 0.05)

    pc_gn = o3d.geometry.PointCloud()
    pc_gn.points = o3d.utility.Vector3dVector(src)
    return pc_gn


if __name__ == "__main__":
    pass
