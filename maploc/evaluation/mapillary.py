# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from maploc import logger
from maploc.conf import data as conf_data_dir
from maploc.data import MapillaryDataModule
from maploc.evaluation.run import evaluate
import numpy as np

split_overrides = {
    "val": {
        "scenes": [
            "sanfrancisco_soma",
            "lemans",
            "toulouse",
            "avignon",
        ],
    },
}
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
# print(f"data_cfg:{data_cfg}")
default_cfg_single = OmegaConf.create({"data": data_cfg})
default_cfg_sequential = OmegaConf.create(
    {
        **default_cfg_single,
        "chunking": {
            "max_length": 10,
        },
    }
)
# print(f"default_cfg_sequential:{default_cfg_sequential}")

def calculate_mean_errors(metrics, thresholds):
    """
    计算位置和方向的均值误差
    
    Args:
        metrics: 评估指标对象
        thresholds: 阈值列表
    
    Returns:
        dict: 包含均值误差的字典
    """
    mean_errors = {}
    
    # 位置误差的键
    position_keys = [
        "xy_max_error",
        "xy_gps_error", 
        "xy_seq_error",
        "xy_gps_seq_error"
    ]
    
    # 方向误差的键
    angle_keys = [
        "yaw_max_error",
        "yaw_seq_error", 
        "yaw_gps_seq_error"
    ]
    
    # 计算位置均值误差
    for key in position_keys:
        if key in metrics:
            errors = metrics[key].get_errors()
            if errors.numel() > 0:
                mean_errors[f"{key}_mean"] = float(errors.mean().item())
                # 计算在不同阈值下的均值误差
                for threshold in thresholds:
                    valid_errors = errors[errors <= threshold]
                    if valid_errors.numel() > 0:
                        mean_errors[f"{key}_mean_under_{threshold}m"] = float(valid_errors.mean().item())
    
    # 计算方向均值误差  
    for key in angle_keys:
        if key in metrics:
            errors = metrics[key].get_errors()
            if errors.numel() > 0:
                mean_errors[f"{key}_mean"] = float(errors.mean().item())
                # 计算在不同阈值下的均值误差
                for threshold in thresholds:
                    valid_errors = errors[errors <= threshold]
                    if valid_errors.numel() > 0:
                        mean_errors[f"{key}_mean_under_{threshold}deg"] = float(valid_errors.mean().item())
    
    return mean_errors

def run(
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int] = (1,2, 3, 5,10),
    **kwargs,
):
    cfg = cfg or {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    default = OmegaConf.merge(default, dict(data=split_overrides[split]))
    cfg = OmegaConf.merge(default, cfg)
    dataset = MapillaryDataModule(cfg.get("data", {}))
    metrics = evaluate(experiment, cfg, dataset, split, sequential=sequential, **kwargs)

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
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    logger.info(OmegaConf.to_yaml(cfg))
    print(f"args message: {args}")
    run(
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        output_dir=args.output_dir,
        num=args.num,
    )
