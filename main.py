#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author awuBugless
@date 2021.10.20
"""

import sys
import copy
import numpy as np
from PyQt5.QtWidgets import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from DeepMerge import merge, del_same_in_src
from DeepMatch import deepmatch_registration
import open3d as o3d


class ui_MainWindow(QMainWindow):
    def __init__(self):
        super(ui_MainWindow, self).__init__()
        self.pcs_before_merge = []
        self.pcs_after_merge = []
        self.vtk_pcs_before_merge = []
        self.vtk_pcs_after_merge = []
        self.vtk_pc_src = vtk.vtkPoints()
        self.match_pc_src = []
        self.vtk_pc_tgt = vtk.vtkPoints()
        self.match_pc_tgt = []
        self.vtk_pc_res = vtk.vtkPoints()
        self.match_pc_res = []
        self.init_main()
        self.init_top()
        self.init_bottom()

    def init_main(self):
        self.setWindowTitle("三维点云数据处理系统")
        self.setGeometry(100, 100, 1200, 680)
        self.setMinimumWidth(1000)
        self.setMinimumHeight(560)

        # 创建主界面
        # 主界面氛围上下两个区域
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

    def init_top(self):
        # 主界面上面的功能区
        # 上面的功能区分为左右两个部分
        self.top_widget = QWidget(self.main_widget)
        self.top_layout = QHBoxLayout(self.top_widget)
        # 将上面的功能区加入 main_layout 中
        self.main_layout.addWidget(self.top_widget)

        # 添加功能选择按钮
        # 点云拼接按钮
        self.push_button_merge = QPushButton()
        self.push_button_merge.setFixedHeight(30)
        self.push_button_merge.setText("点云拼接")
        self.push_button_merge.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_merge.clicked.connect(self.on_push_button_merge_clicked)

        self.top_layout.addWidget(self.push_button_merge)

        # 点云配准按钮
        self.push_button_match = QPushButton()
        self.push_button_match.setFixedHeight(30)
        self.push_button_match.setText("点云配准")
        self.push_button_match.setStyleSheet("background-color:gray")
        # 按钮事件
        self.push_button_match.clicked.connect(self.on_push_button_match_clicked)
        self.top_layout.addWidget(self.push_button_match)

    def init_bottom(self):
        """底部区域"""
        self.bottom_widget = QStackedWidget(self.main_widget)
        self.main_layout.addWidget(self.bottom_widget)

        """设置拼接功能区"""
        self.form_merge = QWidget(self.bottom_widget)
        self.form_merge_layout = QVBoxLayout(self.form_merge)

        # 该区域是功能区
        self.bottom_top_merge_widget = QWidget(self.form_merge)
        self.bottom_top_merge_layout = QHBoxLayout(self.bottom_top_merge_widget)
        self.form_merge_layout.addWidget(self.bottom_top_merge_widget)

        # 输入点云按钮
        self.push_button_merge_open = QPushButton()
        self.push_button_merge_open.setFixedHeight(30)
        self.push_button_merge_open.setText("输入点云")
        self.push_button_merge_open.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_merge_open.clicked.connect(self.open_merge)
        self.bottom_top_merge_layout.addWidget(self.push_button_merge_open)

        # 保存点云按钮
        self.push_button_merge_save_merge = QPushButton()
        self.push_button_merge_save_merge.setFixedHeight(30)
        self.push_button_merge_save_merge.setText("保存")
        self.push_button_merge_save_merge.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_merge_save_merge.clicked.connect(self.save_merge)
        self.bottom_top_merge_layout.addWidget(self.push_button_merge_save_merge)

        # 清除点云按钮
        self.push_button_merge_clear_merge = QPushButton()
        self.push_button_merge_clear_merge.setFixedHeight(30)
        self.push_button_merge_clear_merge.setText("清除")
        self.push_button_merge_clear_merge.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_merge_clear_merge.clicked.connect(self.clear_merge)
        self.bottom_top_merge_layout.addWidget(self.push_button_merge_clear_merge)

        """主界面下面的显示区"""
        # 显示区分为两个显示窗口
        self.bottom_bottom_merge_widget = QWidget(self.form_merge)
        self.bottom_bottom_merge_layout = QHBoxLayout(self.bottom_bottom_merge_widget)
        self.form_merge_layout.addWidget(self.bottom_bottom_merge_widget)

        # 左边的显示窗口
        self.left_merge_widget = QVTKRenderWindowInteractor(self.bottom_bottom_merge_widget)
        self.left_merge_widget.GlobalWarningDisplayOff();
        self.bottom_bottom_merge_layout.addWidget(self.left_merge_widget)
        self.left_merge_disp = vtk.vtkRenderer()
        self.left_merge_disp.SetBackground(.2, .3, .4)
        self.left_merge_widget.GetRenderWindow().AddRenderer(self.left_merge_disp)
        self.left_merge_bk = self.left_merge_widget.GetRenderWindow().GetInteractor()

        # 右边的显示窗口
        self.right_merge_widget = QVTKRenderWindowInteractor(self.bottom_bottom_merge_widget)
        self.right_merge_widget.GlobalWarningDisplayOff();
        self.bottom_bottom_merge_layout.addWidget(self.right_merge_widget)
        self.right_merge_disp = vtk.vtkRenderer()
        self.right_merge_disp.SetBackground(.2, .3, .4)
        self.right_merge_widget.GetRenderWindow().AddRenderer(self.right_merge_disp)
        self.right_merge_bk = self.right_merge_widget.GetRenderWindow().GetInteractor()

        """设置配准功能区"""
        self.form_match = QWidget(self.bottom_widget)
        self.form_match_layout = QVBoxLayout(self.form_match)

        # 该区域是功能区
        self.bottom_top_match_widget = QWidget(self.form_match)
        self.bottom_top_match_layout = QHBoxLayout(self.bottom_top_match_widget)
        self.form_match_layout.addWidget(self.bottom_top_match_widget)

        # 输入源点云按钮
        self.push_button_match_src = QPushButton()
        self.push_button_match_src.setFixedHeight(30)
        self.push_button_match_src.setText("源点云")
        self.push_button_match_src.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_match_src.clicked.connect(self.open_match_src)
        self.bottom_top_match_layout.addWidget(self.push_button_match_src)

        # 输入目标点云按钮
        self.push_button_match_tgt = QPushButton()
        self.push_button_match_tgt.setFixedHeight(30)
        self.push_button_match_tgt.setText("目标点云")
        self.push_button_match_tgt.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_match_tgt.clicked.connect(self.open_match_tgt)
        self.bottom_top_match_layout.addWidget(self.push_button_match_tgt)

        # 保存按钮
        self.push_button_match_reg_match = QPushButton()
        self.push_button_match_reg_match.setFixedHeight(30)
        self.push_button_match_reg_match.setText("配准")
        self.push_button_match_reg_match.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_match_reg_match.clicked.connect(self.reg_match)
        self.bottom_top_match_layout.addWidget(self.push_button_match_reg_match)

        # 保存按钮
        self.push_button_match_save_match = QPushButton()
        self.push_button_match_save_match.setFixedHeight(30)
        self.push_button_match_save_match.setText("保存")
        self.push_button_match_save_match.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_match_save_match.clicked.connect(self.save_match)
        self.bottom_top_match_layout.addWidget(self.push_button_match_save_match)

        # 清除点云按钮
        self.push_button_merge_clear_match = QPushButton()
        self.push_button_merge_clear_match.setFixedHeight(30)
        self.push_button_merge_clear_match.setText("清除")
        self.push_button_merge_clear_match.setStyleSheet("background-color:white")
        # 按钮事件
        self.push_button_merge_clear_match.clicked.connect(self.clear_match)
        self.bottom_top_match_layout.addWidget(self.push_button_merge_clear_match)

        """主界面下面的显示区"""
        # 显示区分为两个显示窗口
        self.bottom_bottom_match_widget = QWidget(self.form_match)
        self.bottom_bottom_match_layout = QHBoxLayout(self.bottom_bottom_match_widget)
        self.form_match_layout.addWidget(self.bottom_bottom_match_widget)

        # 左边的显示窗口
        self.left_match_widget = QVTKRenderWindowInteractor(self.bottom_bottom_match_widget)
        self.left_match_widget.GlobalWarningDisplayOff();
        self.bottom_bottom_match_layout.addWidget(self.left_match_widget)
        self.left_match_disp = vtk.vtkRenderer()
        self.left_match_disp.SetBackground(.2, .3, .4)
        self.left_match_widget.GetRenderWindow().AddRenderer(self.left_match_disp)
        self.left_match_bk = self.left_match_widget.GetRenderWindow().GetInteractor()

        # 右边的显示窗口
        self.right_match_widget = QVTKRenderWindowInteractor(self.bottom_bottom_match_widget)
        self.right_match_widget.GlobalWarningDisplayOff();
        self.bottom_bottom_match_layout.addWidget(self.right_match_widget)
        self.right_match_disp = vtk.vtkRenderer()
        self.right_match_disp.SetBackground(.2, .3, .4)
        self.right_match_widget.GetRenderWindow().AddRenderer(self.right_match_disp)
        self.right_match_bk = self.right_match_widget.GetRenderWindow().GetInteractor()

        """将两个功能区加入 mid_widget"""
        self.bottom_widget.addWidget(self.form_merge)
        self.bottom_widget.addWidget(self.form_match)

    # 按钮一：打开点云拼接面板
    def on_push_button_merge_clicked(self):
        self.push_button_merge.setStyleSheet("background-color:white")
        self.push_button_match.setStyleSheet("background-color:gray")
        self.bottom_widget.setCurrentIndex(0)

    # 按钮二：打开点云配准面板
    def on_push_button_match_clicked(self):
        self.push_button_merge.setStyleSheet("background-color:gray")
        self.push_button_match.setStyleSheet("background-color:white")
        self.bottom_widget.setCurrentIndex(1)

    def open_merge(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择点云", "", "(*.ply *.pcd)")
        if file_name:
            is_readed = False
            for pc_before_merge in self.pcs_before_merge:
                if pc_before_merge[2] == file_name:
                    is_readed = True
            if is_readed:
                QMessageBox.warning(self.form_match, "警告", "请勿重复输入点云！")
            else:
                pc = o3d.io.read_point_cloud(file_name)
                pc_down = o3d.geometry.PointCloud.uniform_down_sample(pc, len(pc.points) // 15000)
                pc_before_merge = []
                pc_before_merge.append(copy.deepcopy(pc))
                pc_before_merge.append(copy.deepcopy(pc_down))
                pc_before_merge.append(file_name)

                pc_after_merge = []
                pc_after_merge.append(copy.deepcopy(pc))
                pc_after_merge.append(copy.deepcopy(pc_down))
                pc_after_merge.append(file_name[:-4] + "_merged" + file_name[-4:])

                is_neared = True

                if len(self.pcs_after_merge) == 0:
                    pc_after_merge.append(np.identity(4))
                    self.pcs_before_merge.append(copy.deepcopy(pc_before_merge))
                    self.pcs_after_merge.append(copy.deepcopy(pc_after_merge))
                else:
                    for pc in self.pcs_after_merge[1:]:
                        pc_after_merge[0].transform(pc[-1])
                    is_neared, tramsformation = merge(pc_after_merge[0], self.pcs_after_merge[-1][0])
                    if is_neared:
                        pc_after_merge[0].transform(tramsformation)
                        pc_after_merge[1].transform(tramsformation)
                        pc_after_merge.append(tramsformation)
                        pc_after_merge[1] = del_same_in_src(pc_after_merge[1], self.pcs_after_merge[-1][1])
                        self.pcs_before_merge.append(copy.deepcopy(pc_before_merge))
                        self.pcs_after_merge.append(copy.deepcopy(pc_after_merge))
                    else:
                        QMessageBox.warning(self.form_match, "警告", "该点云和上一片点云不属于邻近点云，请重新输入！")

                if is_neared:
                    """左边显示拼接前的点云"""
                    vtk_pc_before_merge = vtk.vtkPoints()
                    for point in self.pcs_before_merge[-1][0].points:
                        vtk_pc_before_merge.InsertNextPoint(point)

                    poly_data_before_merge = vtk.vtkPolyData()
                    poly_data_before_merge.SetPoints(vtk_pc_before_merge)

                    glyph_filter_before_merge = vtk.vtkVertexGlyphFilter()
                    glyph_filter_before_merge.SetInputData(poly_data_before_merge)
                    glyph_filter_before_merge.Update()

                    # Create a mapper
                    mapper_before_merge = vtk.vtkPolyDataMapper()
                    mapper_before_merge.SetInputConnection(glyph_filter_before_merge.GetOutputPort())

                    # Create an actor
                    actor_before_merge = vtk.vtkActor()
                    actor_before_merge.SetMapper(mapper_before_merge)
                    if len(self.pcs_before_merge) & 1:
                        actor_before_merge.GetProperty().SetColor(1.0, 0, 0)
                    else:
                        actor_before_merge.GetProperty().SetColor(0, 1.0, 0)

                    self.left_merge_disp.AddActor(actor_before_merge)
                    self.left_merge_disp.ResetCamera()
                    self.vtk_pcs_before_merge.append(vtk_pc_before_merge)

                    """右边显示拼接前的点云"""
                    vtk_pc_after_merge = vtk.vtkPoints()
                    for point in self.pcs_after_merge[-1][1].points:
                        vtk_pc_after_merge.InsertNextPoint(point)

                    poly_data_after_merge = vtk.vtkPolyData()
                    poly_data_after_merge.SetPoints(vtk_pc_after_merge)

                    glyph_filter_after_merge = vtk.vtkVertexGlyphFilter()
                    glyph_filter_after_merge.SetInputData(poly_data_after_merge)
                    glyph_filter_after_merge.Update()

                    # Create a mapper
                    mapper_after_merge = vtk.vtkPolyDataMapper()
                    mapper_after_merge.SetInputConnection(glyph_filter_after_merge.GetOutputPort())

                    # Create an actor
                    actor_after_merge = vtk.vtkActor()
                    actor_after_merge.SetMapper(mapper_after_merge)
                    if len(self.pcs_after_merge) & 1:
                        actor_after_merge.GetProperty().SetColor(1.0, 0, 0)
                    else:
                        actor_after_merge.GetProperty().SetColor(0, 1.0, 0)

                    self.right_merge_disp.AddActor(actor_after_merge)
                    self.right_merge_disp.ResetCamera()
                    self.vtk_pcs_after_merge.append(vtk_pc_after_merge)

    def save_merge(self):
        if not self.pcs_after_merge:
            QMessageBox.warning(self.form_match, "警告", "保存失败，请至少输入一个点云！")
        else:
            points_all = np.asarray(self.pcs_after_merge[0][0].points)
            for pc_after_merge in self.pcs_after_merge[1:]:
                points = np.asarray(pc_after_merge[0].points)
                points_all = np.vstack((points_all, points))
            self.pcs_after_merge[0][0].points = o3d.utility.Vector3dVector(points_all)
            for pc_after_merge in self.pcs_after_merge:
                o3d.io.write_point_cloud(pc_after_merge[2], pc_after_merge[0])
            QMessageBox.about(self.form_merge, "提示", "拼接后的点云已保存在每一个点云同一文件夹下，文件名为每个点云文件名加后缀"
                                                     "\"_merged\"。\n注意：第一个点云保存的是拼接后的完整点云！")

    def clear_merge(self):
        self.pcs_before_merge = []
        self.pcs_after_merge = []
        for vtk_pc_before_merge in self.vtk_pcs_before_merge:
            vtk_pc_before_merge.Reset()
        self.vtk_pcs_before_merge = []
        for vtk_pc_after_merge in self.vtk_pcs_after_merge:
            vtk_pc_after_merge.Reset()
        self.vtk_pcs_after_merge = []
        QMessageBox.about(self.form_merge, "提示", "清除成功!")

    def open_match_src(self):
        if self.match_pc_src:
            QMessageBox.warning(self.form_match, "警告", "请先点击清除键删除之前的数据！")
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择源点云", "", "(*.ply *.pcd)")
            if file_name:
                if self.match_pc_tgt and file_name == self.match_pc_tgt[-1]:
                    QMessageBox.warning(self.form_match, "警告", "和目标点云相同，请重新输入！")
                else:
                    self.match_pc_res = []  # 用于判断是否重复配准

                    self.match_pc_src = []
                    pc = o3d.io.read_point_cloud(file_name)
                    self.match_pc_src.append(copy.deepcopy(pc))
                    self.match_pc_src.append(file_name)
                    """源点云 vtk 显示"""
                    # Create pointcloud
                    for point in self.match_pc_src[0].points:
                        self.vtk_pc_src.InsertNextPoint(point)

                    poly_data_src = vtk.vtkPolyData()
                    poly_data_src.SetPoints(self.vtk_pc_src)

                    glyph_filter_src = vtk.vtkVertexGlyphFilter()
                    glyph_filter_src.SetInputData(poly_data_src)
                    glyph_filter_src.Update()

                    # Create a mapper
                    mapper_src = vtk.vtkPolyDataMapper()
                    mapper_src.SetColorModeToDefault()
                    mapper_src.SetInputConnection(glyph_filter_src.GetOutputPort())

                    # Create an actor
                    actor_src = vtk.vtkActor()
                    actor_src.SetMapper(mapper_src)
                    actor_src.GetProperty().SetColor(1.0, 0, 0)

                    self.left_match_disp.AddActor(actor_src)
                    self.left_match_disp.ResetCamera()

    def open_match_tgt(self):
        if self.match_pc_tgt:
            QMessageBox.warning(self.form_match, "警告", "请先点击清除键删除之前的数据！")
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择目标点云", "", "(*.ply *.pcd)")

            if file_name:
                if self.match_pc_src and file_name == self.match_pc_src[-1]:
                    QMessageBox.warning(self.form_match, "警告", "和源点云相同，请重新输入！")
                else:
                    self.match_pc_res = []  # 用于判断是否重复配准

                    self.match_pc_tgt = []
                    pc = o3d.io.read_point_cloud(file_name)
                    self.match_pc_tgt.append(copy.deepcopy(pc))
                    self.match_pc_tgt.append(file_name)
                    """目标点云 vtk 显示"""
                    # Create pointcloud
                    for point in self.match_pc_tgt[0].points:
                        self.vtk_pc_tgt.InsertNextPoint(point)

                    poly_data_tgt = vtk.vtkPolyData()
                    poly_data_tgt.SetPoints(self.vtk_pc_tgt)

                    glyph_filter_tgt = vtk.vtkVertexGlyphFilter()
                    glyph_filter_tgt.SetInputData(poly_data_tgt)
                    glyph_filter_tgt.Update()

                    # Create a mapper
                    mapper_tgt = vtk.vtkPolyDataMapper()
                    mapper_tgt.SetInputConnection(glyph_filter_tgt.GetOutputPort())

                    # Create an actor
                    actor_tgt = vtk.vtkActor()
                    actor_tgt.SetMapper(mapper_tgt)
                    actor_tgt.GetProperty().SetColor(0, 1.0, 0)

                    self.left_match_disp.AddActor(actor_tgt)
                    self.left_match_disp.ResetCamera()

    def save_match(self):
        if not self.match_pc_res:
            if not self.match_pc_src and not self.match_pc_tgt:
                QMessageBox.warning(self.form_match, "警告", "保存失败，请输入源点云和目标点云并完成配准后再尝试！")
            elif not self.match_pc_src:
                QMessageBox.warning(self.form_match, "警告", "保存失败，请输入源点云并完成配准后再尝试！")
            elif not self.match_pc_tgt:
                QMessageBox.warning(self.form_match, "警告", "保存失败，请输入目标点云并完成配准后再尝试！")
            else:
                QMessageBox.warning(self.form_match, "警告", "保存失败，请点击配准后再尝试！")
        else:
            o3d.io.write_point_cloud(self.match_pc_res[1], self.match_pc_res[0])
            QMessageBox.about(self.form_match, "提示", "配准后的点云已保存在源点云同一文件夹下，文件名为源点云文件名加后缀\"_matched\"")

    def clear_match(self):
        self.match_pc_src = []
        self.match_pc_tgt = []
        self.match_pc_res = []
        """在此处添加清除vtk中显示的代码"""
        self.vtk_pc_src.Reset()
        self.vtk_pc_tgt.Reset()
        self.vtk_pc_res.Reset()
        QMessageBox.about(self.form_match, "提示", "清除成功!")

    def reg_match(self):
        if self.match_pc_res:
            QMessageBox.warning(self.form_match, "警告", "已完成配准，请勿重复点击！")
        if not self.match_pc_src and not self.match_pc_tgt:
            QMessageBox.warning(self.form_match, "警告", "配准失败，请输入源点云和目标点云后再尝试！")
        elif not self.match_pc_src:
            QMessageBox.warning(self.form_match, "警告", "配准失败，请输入源点云再尝试！")
        elif not self.match_pc_tgt:
            QMessageBox.warning(self.form_match, "警告", "配准失败，请输入目标点云再尝试！")
        else:
            self.match_pc_res.append(copy.deepcopy(self.match_pc_src[0]))
            self.match_pc_res.append(self.match_pc_src[1][:-4] + "_matched" + self.match_pc_src[1][-4:])
            transformation = deepmatch_registration(self.match_pc_src[0], self.match_pc_tgt[0])
            self.match_pc_res[0].transform(transformation)
            """结果点云 vtk 显示"""
            # Create pointcloud
            for point in self.match_pc_res[0].points:
                self.vtk_pc_res.InsertNextPoint(point)

            poly_data_res = vtk.vtkPolyData()
            poly_data_res.SetPoints(self.vtk_pc_res)

            glyph_filter_res = vtk.vtkVertexGlyphFilter()
            glyph_filter_res.SetInputData(poly_data_res)
            glyph_filter_res.Update()

            # Create a mapper
            mapper_res = vtk.vtkPolyDataMapper()
            mapper_res.SetInputConnection(glyph_filter_res.GetOutputPort())

            # Create an actor
            actor_res = vtk.vtkActor()
            actor_res.SetMapper(mapper_res)
            actor_res.GetProperty().SetColor(1.0, 0, 0)

            self.right_match_disp.AddActor(actor_res)
            self.right_match_disp.ResetCamera()

            """目标点云 vtk 显示"""
            # Create pointcloud
            poly_data_tgt = vtk.vtkPolyData()
            poly_data_tgt.SetPoints(self.vtk_pc_tgt)

            glyph_filter_tgt = vtk.vtkVertexGlyphFilter()
            glyph_filter_tgt.SetInputData(poly_data_tgt)
            glyph_filter_tgt.Update()

            # Create a mapper
            mapper_tgt = vtk.vtkPolyDataMapper()
            mapper_tgt.SetInputConnection(glyph_filter_tgt.GetOutputPort())

            # Create an actor
            actor_tgt = vtk.vtkActor()
            actor_tgt.SetMapper(mapper_tgt)
            actor_tgt.GetProperty().SetColor(0, 1.0, 0)

            self.right_match_disp.AddActor(actor_tgt)
            self.right_match_disp.ResetCamera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    mainWindow = ui_MainWindow()
    mainWindow.show()
    mainWindow.left_merge_bk.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    mainWindow.left_merge_bk.Initialize()
    mainWindow.left_merge_bk.Start()
    mainWindow.right_merge_bk.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    mainWindow.right_merge_bk.Initialize()
    mainWindow.right_merge_bk.Start()
    mainWindow.left_match_bk.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    mainWindow.left_match_bk.Initialize()
    mainWindow.left_match_bk.Start()
    mainWindow.right_match_bk.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    mainWindow.right_match_bk.Initialize()
    mainWindow.right_match_bk.Start()
    sys.exit(app.exec_())
