import fnmatch
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from maploc.utils.io import read_image
import utm
import math


# 计算两个点之间的欧几里得距离
def haversine_distance(prior_latlon, actual_latlon):
    # 将经纬度转换为 UTM 坐标
    utm1 = utm.from_latlon(actual_latlon[0], actual_latlon[1])
    utm2 = utm.from_latlon(prior_latlon[0], prior_latlon[1])

    # 提取 UTM 坐标（北坐标和东坐标）
    x1, y1 = utm1[0], utm1[1]
    x2, y2 = utm2[0], utm2[1]

    # 计算欧几里得距离
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance



def select_image_file():
    """弹出文件选择框选择图片"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择图片文件",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG文件", "*.jpg *.jpeg"),
            ("PNG文件", "*.png"),
            ("所有文件", "*.*")
        ]
    )

    root.destroy()
    return file_path

def select_image_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择包含图像的文件夹")
    return folder_path if folder_path else None


def get_pose_info_from_json(image_name, pose_json_path="D:/SoftWare/Desktop/OrienterNet/Mychange/pose.json", posecls="initial_pose"):
    """从pose.json中获取图片的位姿信息"""
    try:
        with open(pose_json_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)

        # 查找匹配的图片记录
        for record in pose_data:
            if record['image_name'] == image_name:
                return record[posecls]['latlon']

        print(f"未找到图片 {image_name} 的位姿信息")
        return None

    except FileNotFoundError:
        print(f"位姿文件不存在: {pose_json_path}")
        return None
    except Exception as e:
        print(f"读取位姿文件错误: {e}")
        return None


def get_pose_info_from_json1(image_name, pose_json_path=None,
                            posecls="ground_truth"):
    """从pose.json中获取图片的位姿信息"""
    try:
        # 生成json文件名的部分匹配模式
        search_pattern = f"*{image_name}_results.json"  # Matches any file containing image_name
        matched_files = [f for f in os.listdir(pose_json_path) if fnmatch.fnmatch(f, search_pattern)]

        if not matched_files:
            print(f"未找到包含 {image_name} 的位姿文件")
            return None
        matched_file = matched_files[0]
        matched_file_path = os.path.join(pose_json_path, matched_file)

        # Print the matched file for debugging
        print(f"找到匹配的位姿文件: {matched_file_path}")

        with open(matched_file_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)

        # 查找匹配的图片记录
        if "image_info" in pose_data and pose_data["image_info"].get("name") == image_name:
            # 根据posecls选择返回的字段
            if posecls == "ground_truth":
                latlon = pose_data.get("ground_truth", {}).get("latlon_converted", None)
            elif posecls == "predictions":
                latlon = pose_data.get("predictions", {}).get("latlon_max_converted", None)
            else:
                latlon = None

            if latlon:
                return latlon
            else:
                print(f"未找到图片 {image_name} 的 {posecls} 位姿信息")
                return None
        else:
            print(f"未找到图片 {image_name} 的位置信息")
            return None
    except Exception as e:
        print(f"读取位姿文件错误: {e}")
        return None

def show_osm_image(image_path, image_name):
    """显示对应的OSM PNG图片"""
    # 从图片路径构建OSM路径
    image_path_obj = Path(image_path)

    # 找到MGL开始的路径部分
    parts = image_path_obj.parts
    mgl_index = -1
    for i, part in enumerate(parts):
        if part == "MGL":
            mgl_index = i
            break

    if mgl_index == -1:
        print("图片路径中未找到MGL文件夹")
        return None

    # 重构OSM路径：MGL/scene/images -> MGL/scene/images_osm
    mgl_parts = parts[mgl_index:]  # 从MGL开始的部分
    osm_parts = list(mgl_parts)
    total_parts = list(mgl_parts)
    # 找到images文件夹并替换为images_osm
    for i, part in enumerate(osm_parts):
        if part == "images":
            osm_parts[i] = "osm_maps"
            total_parts[i] = "total"
            break

    # 构建完整的OSM路径
    osm_path = Path("D:/SoftWare/Desktop/PaperCode") / Path(*osm_parts[:-1]) / f"{image_name}.png"  # 去掉原文件名，加上新的png文件名
    total_path = Path("D:/SoftWare/Desktop/PaperCode") / Path(*total_parts[:-1]) /f"{image_name}.png"
    #print(f"构建的OSM路径: {osm_path}")
    #print(f"构建的三图整体路径: {total_path}")
    if osm_path.exists():
        #print(f"找到OSM地图: {osm_path}")
        # # 使用PIL加载并显示OSM图片
        # osm_img = Image.open(osm_path)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(osm_img)
        # plt.title(f"OSM Map for {image_name}")
        # plt.axis('off')
        # plt.show()
        image=read_image(osm_path)
        return image, total_path
    else:
        print(f"未找到OSM地图文件: {osm_path}")
        return None
