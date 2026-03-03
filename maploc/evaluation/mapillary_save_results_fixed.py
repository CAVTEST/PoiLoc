# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch

from omegaconf import DictConfig, OmegaConf

from Mychange.poi_localization_pipeline import POILocalizationPipeline
from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryDataModule
from .run import evaluate, evaluate_single_image, evaluate_sequential
from .utils import write_dump


def save_results_callback(output_dir: Path, tile_managers: dict, save_format: str = "json"):
    """
    创建一个回调函数，用于保存每一张图像的评估结果
    """
    def callback(idx, model, pred, data, results, *args):
        # 处理单个图像的情况
        if isinstance(data, dict):
            data = [data]
            pred = [pred]
            results = [results]
        
        for i, (single_data, single_pred, single_results) in enumerate(zip(data, pred, results)):

            # 获取基本信息
            scene = single_data["scene"]
            name = single_data["name"]
            sequence = single_data.get("sequence", "")
            print(f"{name}")
            # 处理数据类型转换的辅助函数
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                elif isinstance(x, list):
                    return np.array(x)
                else:
                    return x
            
            # 转换为JSON可序列化的格式
            def to_json_safe(x):
                if x is None:
                    return None
                if isinstance(x, np.ndarray):
                    return [float(item) for item in x.flatten()]
                elif isinstance(x, (np.integer, np.int64, np.int32)):
                    return int(x)
                elif isinstance(x, (np.floating, np.float64, np.float32)):
                    return float(x)
                elif isinstance(x, torch.Tensor):
                    return [float(item) for item in x.cpu().numpy().flatten()]
                elif isinstance(x, list):
                    return [to_json_safe(item) for item in x]
                else:
                    return x
            
            # 真值（Ground Truth）
            uv_gt = to_numpy(single_data["uv"])  # 图像坐标系
            yaw_gt = to_numpy(single_data["roll_pitch_yaw"][-1])  # 角度（度）
            
            # 获取canvas用于坐标转换
            canvas = single_data["canvas"]
            # 如果canvas是列表，取第一个元素
            if isinstance(canvas, list):
                canvas = canvas[0]
            
            # 世界坐标系真值
            xy_gt_world = canvas.to_xy(uv_gt)  # 米为单位
            
            # 地理坐标系真值（如果有GPS信息）
            latlon_gt = None
            if "gps_position" in single_data:
                latlon_gt = to_numpy(single_data["gps_position"][:2])
            
            # 预测值
            uv_pred = to_numpy(single_pred["uv_max"])  # 图像坐标系
            yaw_pred = to_numpy(single_pred["yaw_max"])  # 角度（度）
            
            # 世界坐标系预测值
            xy_pred_world = canvas.to_xy(uv_pred)  # 米为单位
            
            # 获取投影对象用于GPS坐标转换
            projection = None
            if scene in tile_managers:
                projection = tile_managers[scene].projection
            elif hasattr(canvas, 'projection') and canvas.projection is not None:
                projection = canvas.projection
            elif "projection" in single_data:
                projection = single_data["projection"]
            
            # GPS坐标转换
            latlon_gt_converted = None
            latlon_pred_converted = None
            latlon_expectation_converted = None

            poi_xypred = None
            poi_gt = None
            if projection is not None:
                # 将世界坐标转换为GPS经纬度
                try:
                    latlon_gt_converted = projection.unproject(xy_gt_world.reshape(1, -1))[0]
                    latlon_pred_converted = projection.unproject(xy_pred_world.reshape(1, -1))[0]
                    logger.warning(f"Failed to convert coordinates to GPS: {e}")
                except Exception as e:
                    logger.warning(f"Failed to convert coordinates to GPS: {e}")
            
            # 期望值（如果有）
            uv_expectation = None
            yaw_expectation = None
            xy_expectation_world = None
            if "uv_expectation" in single_pred:
                uv_expectation = to_numpy(single_pred["uv_expectation"])
                yaw_expectation = to_numpy(single_pred["yaw_expectation"])
                # 计算期望值的世界坐标和GPS坐标
                if uv_expectation is not None:
                    xy_expectation_world = canvas.to_xy(uv_expectation)
                    if projection is not None:
                        try:
                            latlon_expectation_converted = projection.unproject(xy_expectation_world.reshape(1, -1))[0]
                        except Exception as e:
                            logger.warning(f"Failed to convert expectation coordinates to GPS: {e}")
            
            # GPS融合结果（如果有）
            uv_gps = None
            uv_fused = None
            yaw_fused = None
            xy_gps_world = None
            xy_fused_world = None
            latlon_gps_converted = None
            latlon_fused_converted = None
            
            if "uv_gps" in single_data:
                uv_gps = to_numpy(single_data["uv_gps"])
                xy_gps_world = canvas.to_xy(uv_gps)
                if projection is not None:
                    try:
                        latlon_gps_converted = projection.unproject(xy_gps_world.reshape(1, -1))[0]
                    except Exception as e:
                        logger.warning(f"Failed to convert GPS coordinates: {e}")
                        
            if "uv_fused" in single_pred:
                uv_fused = to_numpy(single_pred["uv_fused"])
                yaw_fused = to_numpy(single_pred["yaw_fused"])
                xy_fused_world = canvas.to_xy(uv_fused)
                if projection is not None:
                    try:
                        latlon_fused_converted = projection.unproject(xy_fused_world.reshape(1, -1))[0]
                    except Exception as e:
                        logger.warning(f"Failed to convert fused coordinates: {e}")
            
            # 构建结果字典
            result_dict = {
                "image_info": {
                    "scene": scene,
                    "sequence": sequence,
                    "name": name,
                    "index": int(idx),
                    "sub_index": int(i)
                },
                "ground_truth": {
                    "uv": to_json_safe(uv_gt),  # 图像坐标系 [u, v]
                    "xy_world": to_json_safe(xy_gt_world),  # 世界坐标系 [x, y] (米)
                    "yaw_degrees": float(yaw_gt),  # 偏航角 (度)
                    "latlon": to_json_safe(latlon_gt) if latlon_gt is not None else None,  # 原始GPS坐标 [lat, lon]
                    "latlon_converted": to_json_safe(latlon_gt_converted) if latlon_gt_converted is not None else None,  # 从世界坐标转换的GPS坐标 [lat, lon]
                    "camera_height": float(single_data.get("camera_height", 0)) if "camera_height" in single_data else None,
                    "poi_gt":to_json_safe(poi_gt),
                },
                "predictions": {
                    "uv_max": to_json_safe(uv_pred),  # 最大概率预测 [u, v]
                    "xy_world_max": to_json_safe(xy_pred_world),  # 世界坐标系预测 [x, y] (米)
                    "yaw_max_degrees": float(yaw_pred),  # 偏航角预测 (度)
                    "latlon_max_converted": to_json_safe(latlon_pred_converted) if latlon_pred_converted is not None else None,  # 预测位置的GPS坐标 [lat, lon]
                    "uv_expectation": to_json_safe(uv_expectation) if uv_expectation is not None else None,
                    "xy_world_expectation": to_json_safe(xy_expectation_world) if xy_expectation_world is not None else None,  # 期望值世界坐标 [x, y] (米)
                    "yaw_expectation_degrees": float(yaw_expectation) if yaw_expectation is not None else None,
                    "poi_xypred": to_json_safe(poi_xypred),
                    "latlon_expectation_converted": to_json_safe(latlon_expectation_converted) if latlon_expectation_converted is not None else None,  # 期望值GPS坐标 [lat, lon]
                },
                "gps_fusion": {
                    "uv_gps": to_json_safe(uv_gps) if uv_gps is not None else None,
                    "xy_world_gps": to_json_safe(xy_gps_world) if xy_gps_world is not None else None,  # GPS世界坐标 [x, y] (米)
                    "latlon_gps_converted": to_json_safe(latlon_gps_converted) if latlon_gps_converted is not None else None,  # GPS转换的经纬度 [lat, lon]
                    "uv_fused": to_json_safe(uv_fused) if uv_fused is not None else None,
                    "xy_world_fused": to_json_safe(xy_fused_world) if xy_fused_world is not None else None,  # 融合世界坐标 [x, y] (米)
                    "latlon_fused_converted": to_json_safe(latlon_fused_converted) if latlon_fused_converted is not None else None,  # 融合转换的经纬度 [lat, lon]
                    "yaw_fused_degrees": float(yaw_fused) if yaw_fused is not None else None
                },
                "metrics": {
                    "xy_error_meters": float(single_results["xy_max_error"]),
                    "yaw_error_degrees": float(single_results["yaw_max_error"]) if "yaw_max_error" in single_results else None
                },
                "map_info": {
                    "pixel_per_meter": float(canvas.ppm),
                    "bbox_min": to_json_safe(canvas.bbox.min_),
                    "bbox_max": to_json_safe(canvas.bbox.max_),
                    "image_size": [int(canvas.w), int(canvas.h)]
                }
            }
            
            # 保存到JSON文件
            if save_format == "json":
                filename = f"{scene}_{sequence}_{name}_results.json"
                filepath = output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved results for {scene}/{sequence}/{name} to {filepath}")
    
    return callback


def run_with_save(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int] = (1, 2, 3, 5,10),
    output_dir: Optional[Path] = None,
    save_format: str = "json",
    **kwargs,
):
    """
    运行评估并保存详细结果
    """
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    
    # 加载默认配置
    data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / "mapillary.yaml")
    data_cfg = OmegaConf.merge(
        data_cfg_train,
        {
            "return_gps": True,
            "add_map_mask": True,
            "max_init_error": 32,
            "loading": {"val": {"batch_size": 1, "num_workers": 0}},
        },
    )
    default_cfg_single = OmegaConf.create({"data": data_cfg})
    default_cfg_sequential = OmegaConf.create(
        {
            **default_cfg_single,
            "chunking": {
                "max_length": 10,
            },
        }
    )
    
    default = default_cfg_sequential if sequential else default_cfg_single
    default = OmegaConf.merge(default, dict(data=split_overrides[split]))
    cfg = OmegaConf.merge(default, cfg)
    
    dataset = MapillaryDataModule(cfg.get("data", {}))
    
    # 准备数据集并设置
    dataset.prepare_data()
    dataset.setup()
    
    # 创建输出目录
    if output_dir is None:
        output_dir = Path(f"mapillary_results_{experiment}_{split}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建保存结果的回调函数
    save_callback = save_results_callback(output_dir, dataset.tile_managers, save_format)
    
    # 运行评估
    metrics = evaluate(
        experiment, 
        cfg, 
        dataset, 
        split, 
        sequential=sequential, 
        output_dir=output_dir,
        callback=save_callback,
        **kwargs
    )
    
    # 保存汇总结果
    keys = [
        "xy_max_error",
        "xy_gps_error",
        "yaw_max_error",
    ]
    if sequential:
        keys += [
            "xy_seq_error",
            "xy_gps_seq_error",
            "yaw_seq_error",
            "yaw_gps_seq_error",
        ]
    
    summary_results = {}
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        summary_results[k] = {
            "recall": rec,
            "thresholds": list(thresholds)
        }
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    
    # 保存汇总结果
    summary_file = output_dir / "summary_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary results saved to {summary_file}")
    logger.info(f"Detailed results saved to {output_dir}")
    
    return metrics


# 分割配置
split_overrides = {
    "val": {
        "scenes": [
            "avignon",
            # "sanfrancisco_soma",
            # "lemans",
            # "berlin"
            # "toulouse"
        ],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mapillary dataset and save detailed results")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name or checkpoint path")
    parser.add_argument("--split", type=str, default="val", choices=["val"], help="Dataset split")
    parser.add_argument("--sequential", action="store_true", help="Use sequential evaluation")
    parser.add_argument("--output_dir", type=Path, help="Output directory for results")
    parser.add_argument("--save_format", type=str, default="json", choices=["json"], help="Output format")
    parser.add_argument("--num", type=int, help="Number of samples to evaluate")
    parser.add_argument("dotlist", nargs="*", help="Additional configuration overrides")
    
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    
    run_with_save(
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        output_dir=args.output_dir,
        save_format=args.save_format,
        num=args.num,
    )
