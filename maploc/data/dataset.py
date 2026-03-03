# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import json
import os

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf

from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
        # pose info saving
        "save_pose_info": True,
        "pose_info_file": "pose_info.json",
        "save_pose_immediately": True,
        "pose_save_interval": 10,
    }

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

        # Initialize pose info collection
        self.pose_info_data = [] if self.cfg.save_pose_info else None
        self.pose_save_counter = 0  # 计数器，用于批量保存

        # Create output directory if needed
        if self.cfg.save_pose_info:
            self.pose_info_path = Path(self.cfg.pose_info_file)
            self.pose_info_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果是立即保存模式，初始化JSON文件
            if self.cfg.save_pose_immediately:
                if self.pose_info_path.exists():
                    # 如果文件已存在，加载现有数据
                    try:
                        with open(self.pose_info_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        print(f"已加载现有位姿信息: {len(existing_data)} 条记录")
                    except:
                        # 文件损坏或格式错误，重新开始
                        with open(self.pose_info_path, 'w', encoding='utf-8') as f:
                            json.dump([], f)
                        print(f"重新初始化位姿信息文件: {self.pose_info_path}")
                else:
                    # 创建新文件
                    with open(self.pose_info_path, 'w', encoding='utf-8') as f:
                        json.dump([], f)
                    print(f"创建位姿信息文件: {self.pose_info_path}")

        tfs = []
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)
        scene, seq, name = self.names[idx]
        print(f"scene, seq, name:{scene},{seq},{name}")
        if self.cfg.init_from_gps:
            print(f"self.cfg.init_from_gps :{self.cfg.init_from_gps}")
            latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            print(f"self.cfg.init_from_gps :{self.cfg.init_from_gps}")
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()
            print(f"xy_w_init :{xy_w_init}")

        if "shifts" in self.data:
            print(f"shifts :True")
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            print(f"shifts :False")
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
            print(f"error :{error}")
        xy_w_init += error * self.cfg.max_init_error
        print(f"xy_w_init,crop_size_meters :{xy_w_init},{self.cfg.crop_size_meters}")
        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        projection = self.tile_managers[scene].projection
        latlon_init = projection.unproject(xy_w_init.reshape(1,-1))[0]
        print(f"bbox_tile :{bbox_tile}")
        print(f"初始位置的经纬度: {latlon_init[0]:.15f}, {latlon_init[1]:.15f}")
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_dict(cam_dict).float()
        print(f"cam_dict ,cam:{cam_dict},{cam}")
        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
            print(f"roll_pitch_yaw true:{roll}, {pitch}, {yaw}")
        else:
            roll, pitch, yaw = decompose_rotmat(self.data["R_c2w"][idx].numpy())
            print(f"roll_pitch_yaw:{roll}, {pitch}, {yaw}")
        print(f"image_dirs path:{self.image_dirs[scene] / (name + self.image_ext)}")
        image = read_image(self.image_dirs[scene] / (name + self.image_ext))

        if "plane_params" in self.data:
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )
        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()
        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        xy_w_gt = self.data["t_c2w"][idx][:2].numpy()
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)

        raster = canvas.raster  # C, H, W

        # Map augmentations
        heading = np.deg2rad(90 - yaw)  # fixme
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
            if self.cfg.augmentation.flip:
                image, raster, uv_gt, heading = random_flip(
                    image, raster, uv_gt, heading, seed
                )
        yaw = 90 - np.rad2deg(heading)  # fixme

        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = yaw + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            xy_gps = self.tile_managers[scene].projection.project(gps)
            data["uv_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        # Add the missing data for the POI pipeline
        data['path'] = str(self.image_dirs[scene] / (name + self.image_ext))
        data['proj'] = self.tile_managers[scene].projection
        data['tiler'] = self.tile_managers[scene]
        data['bbox'] = bbox_tile
        data['xy_w_gt'] = xy_w_gt
        data['yaw_gt'] = yaw
        # 保存位姿信息（在所有处理完成后）
        self._save_pose_info(idx, scene, seq, name, bbox_tile.center, xy_w_gt, bbox_tile, yaw)

        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "uv": torch.from_numpy(uv_gt).float(),
            "uv_init": torch.from_numpy(uv_init).float(),
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, roll, pitch, seed):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        image, valid = rectify_image(
            image, cam, roll, pitch if self.cfg.rectify_pitch else None
        )
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask

    def _save_pose_info(self, idx, scene, seq, name, xy_w_init, xy_w_gt, bbox_tile, yaw):
        """收集并保存位姿信息"""
        if not self.cfg.save_pose_info:
            return

        try:
            # 获取投影器
            projection = self.tile_managers[scene].projection

            # 转换坐标到经纬度
            # 初始位姿（加了误差的位置）
            latlon_init = projection.unproject(xy_w_init.reshape(1, -1))[0]

            # 真值位姿
            latlon_gt = projection.unproject(xy_w_gt.reshape(1, -1))[0]

            # 地图边界框转换到经纬度
            bbox_latlon = projection.unproject(bbox_tile)

            # 获取图像路径
            image_path = str(self.image_dirs[scene] / (name + self.image_ext))

            # GPS信息（如果有的话）
            gps_info = {}
            if "gps_position" in self.data and idx < len(self.data["gps_position"]):
                gps_pos = self.data["gps_position"][idx]
                if len(gps_pos) >= 2:
                    gps_info = {
                        "latitude": float(gps_pos[0]),
                        "longitude": float(gps_pos[1])
                    }
                    if len(gps_pos) >= 3:
                        gps_info["altitude"] = float(gps_pos[2])

            # 姿态信息
            roll, pitch = None, None
            if "roll_pitch_yaw" in self.data and idx < len(self.data["roll_pitch_yaw"]):
                rpy = self.data["roll_pitch_yaw"][idx]
                roll = float(rpy[0])
                pitch = float(rpy[1])
                # yaw 已经作为参数传入

            # 构建位姿信息记录
            pose_info = {
                "index": int(idx),
                "image_name": name,
                "scene": scene,
                "sequence": seq,
                "image_path": image_path,

                # 初始位姿（加了误差的位置）
                "initial_pose": {
                    "xy_world": [float(xy_w_init[0]), float(xy_w_init[1])],
                    "latlon": [float(latlon_init[0]), float(latlon_init[1])],
                    "yaw_degrees": float(yaw),
                    "bbox_world": {
                        "min": [float(bbox_tile.min_[0]), float(bbox_tile.min_[1])],
                        "max": [float(bbox_tile.max_[0]), float(bbox_tile.max_[1])]
                    },
                    "bbox_latlon": {
                        "min": [float(bbox_latlon.min_[0]), float(bbox_latlon.min_[1])],
                        "max": [float(bbox_latlon.max_[0]), float(bbox_latlon.max_[1])]
                    }
                },

                # 真值位姿
                "ground_truth": {
                    "xy_world": [float(xy_w_gt[0]), float(xy_w_gt[1])],
                    "latlon": [float(latlon_gt[0]), float(latlon_gt[1])],
                    "roll_degrees": roll,
                    "pitch_degrees": pitch,
                    "yaw_degrees": float(yaw)  # 注意：这里的yaw可能需要调整，因为真值和初始值可能不同
                },

                # GPS信息
                "gps_info": gps_info,

                # 配置信息
                "config": {
                    "max_init_error": float(self.cfg.max_init_error),
                    "crop_size_meters": float(self.cfg.crop_size_meters),
                    "pixel_per_meter": float(self.cfg.pixel_per_meter),
                    "init_from_gps": bool(self.cfg.init_from_gps)
                }
            }

            # 根据配置选择保存策略
            if self.cfg.save_pose_immediately:
                # 立即保存模式：直接追加到文件
                self._append_pose_info_to_file(pose_info)
            else:
                # 批量保存模式：先存储到内存
                self.pose_info_data.append(pose_info)
                self.pose_save_counter += 1

                # 达到保存间隔时批量保存
                if self.pose_save_counter >= self.cfg.pose_save_interval:
                    self.save_pose_info_to_file()
                    self.pose_info_data.clear()  # 清空已保存的数据
                    self.pose_save_counter = 0

        except Exception as e:
            print(f"Warning: Failed to save pose info for {name}: {e}")

    def save_pose_info_to_file(self, output_path=None):
        """将收集的位姿信息保存到JSON文件"""
        if not self.cfg.save_pose_info or not self.pose_info_data:
            return

        if output_path is None:
            output_path = self.cfg.pose_info_file

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.pose_info_data, f, indent=2, ensure_ascii=False)
            print(f"位姿信息已保存到: {output_path} ({len(self.pose_info_data)} 条记录)")
        except Exception as e:
            print(f"保存位姿信息失败: {e}")

    def _append_pose_info_to_file(self, pose_info):
        """立即将单条位姿信息追加到JSON文件"""
        try:
            # 读取现有数据
            if self.pose_info_path.exists():
                with open(self.pose_info_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # 添加新记录
            existing_data.append(pose_info)

            # 写回文件
            with open(self.pose_info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            # 每10条记录打印一次进度
            if len(existing_data) % 10 == 0:
                print(f"已保存位姿信息: {len(existing_data)} 条记录")

        except Exception as e:
            print(f"立即保存位姿信息失败: {e}")

    def __del__(self):
        if hasattr(self, 'cfg') and self.cfg.save_pose_info and not self.cfg.save_pose_immediately:
            if hasattr(self, 'pose_info_data') and self.pose_info_data:
                self.save_pose_info_to_file()

