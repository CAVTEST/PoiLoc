# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import asyncio
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from opensfm.pygeometry import Camera
from opensfm.pymap import Shot
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ... import logger
from ...osm.tiling import TileManager
from ...osm.viz import GeoPlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import DATA_URL, download_file, write_json
from ..utils import decompose_rotmat
from .dataset import MapillaryDataModule
from .download import (
    MapillaryDownloader,
    fetch_image_infos,
    fetch_images_pixels,
    image_filename,
    opensfm_shot_from_info,
)
from .utils import (
    CameraUndistorter,
    PanoramaUndistorter,
    keyframe_selection,
    perspective_camera_from_pano,
    scale_camera,
    undistort_shot,
)

location_to_params = {
    "sanfrancisco_soma": {
        "bbox": BoundaryBox(
            [-122.410307, 37.770364][::-1], [-122.388772, 37.795545][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "osm_file": "sanfrancisco.osm",
    },
    "sanfrancisco_hayes": {
        "bbox": BoundaryBox(
            [-122.438415, 37.768634][::-1], [-122.410605, 37.783894][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "osm_file": "sanfrancisco.osm",
    },
    "amsterdam": {
        "bbox": BoundaryBox([4.845284, 52.340679][::-1], [4.926147, 52.386299][::-1]),
        "camera_models": ["GoPro Max"],
        "osm_file": "amsterdam.osm",
    },#no data
    "lemans": {
        "bbox": BoundaryBox([0.185752, 47.995125][::-1], [0.224088, 48.014209][::-1]),
        "owners": ["xXOocM1jUB4jaaeukKkmgw"],  # sogefi
        "osm_file": "lemans.osm",
    },
    "berlin": {
        "bbox": BoundaryBox([13.416271, 52.459656][::-1], [13.469829, 52.499195][::-1]),
        "owners": ["LT3ajUxH6qsosamrOHIrFw"],  # supaplex030
        "osm_file": "berlin.osm",
    },
    "montrouge": {
        "bbox": BoundaryBox([2.298958, 48.80874][::-1], [2.332989, 48.825276][::-1]),
        "owners": [
            "XtzGKZX2_VIJRoiJ8IWRNQ",
            "C4ENdWpJdFNf8CvnQd7NrQ",
            "e_ZBE6mFd7CYNjRSpLl-Lg",
        ],  # overflorian, phyks, francois2
        "camera_models": ["LG-R105"],
        "osm_file": "paris.osm",
    },
    "nantes": {
        "bbox": BoundaryBox([-1.585839, 47.198289][::-1], [-1.51318, 47.236161][::-1]),
        "owners": [
            "jGdq3CL-9N-Esvj3mtCWew",
            "s-j5BH9JRIzsgORgaJF3aA",
        ],  # c_mobilite, cartocite
        "osm_file": "nantes.osm",
    },
    "toulouse": {
        "bbox": BoundaryBox([1.429457, 43.591434][::-1], [1.456653, 43.61343][::-1]),
        "owners": ["MNkhq6MCoPsdQNGTMh3qsQ"],  # tyndare
        "osm_file": "toulouse.osm",
    },
    "vilnius": {
        "bbox": BoundaryBox([25.258633, 54.672956][::-1], [25.296094, 54.696755][::-1]),
        "owners": ["bClduFF6Gq16cfwCdhWivw", "u5ukBseATUS8jUbtE43fcO"],  # kedas, vms
        "osm_file": "vilnius.osm",
    },
    "helsinki": {
        "bbox": BoundaryBox(
            [24.8975480117, 60.1449128318][::-1], [24.9816543235, 60.1770977471][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "helsinki.osm",
    },
    "milan": {
        "bbox": BoundaryBox(
            [9.1732723899, 45.4810977947][::-1],
            [9.2255987917, 45.5284238563][::-1],
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "milan.osm",
    },
    "avignon": {
        "bbox": BoundaryBox(
            [4.7887045302, 43.9416178156][::-1], [4.8227015622, 43.9584848909][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "avignon.osm",
    },
    "paris": {
        "bbox": BoundaryBox([2.306823, 48.833827][::-1], [2.39067, 48.889335][::-1]),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "paris.osm",
    },
}

# 默认配置，使用 OmegaConf 方便命令行或文件覆盖
default_cfg = OmegaConf.create(
    {
        "max_image_size": 512,  # 处理图像的最大边长尺寸
        "do_legacy_pano_offset": True,  # 是否采用历史偏移方案计算全景图偏移
        "min_dist_between_keyframes": 4,  # 关键帧间的最小距离（米）
        "tiling": {  # 地图瓦片相关配置
            "tile_size": 128,  # 单个瓦片大小（像素）
            "margin": 128,  # 瓦片边缘的扩展margin（像素）
            "ppm": 2,  # 瓦片每米像素数（pixel per meter）
        },
    }
)

def get_pano_offset(image_info: dict, do_legacy: bool = False) -> float:
    if do_legacy:
        seed = int(image_info["sfm_cluster"]["id"])
    else:
        seed = image_info["sequence"].__hash__()
    seed = seed % (2**32 - 1)
    return np.random.RandomState(seed).uniform(-45, 45)


def process_shot(
    shot: Shot, info: dict, image_path: Path, output_dir: Path, cfg: DictConfig
) -> List[Shot]:
    if not image_path.exists():
        return None

    image_orig = cv2.imread(str(image_path))
    max_size = cfg.max_image_size
    pano_offset = None

    camera = shot.camera
    camera.width, camera.height = image_orig.shape[:2][::-1]
    if camera.is_panorama(camera.projection_type):
        camera_new = perspective_camera_from_pano(camera, max_size)
        undistorter = PanoramaUndistorter(camera, camera_new)
        pano_offset = get_pano_offset(info, cfg.do_legacy_pano_offset)
    elif camera.projection_type in ["fisheye", "perspective"]:
        if camera.projection_type == "fisheye":
            camera_new = perspective_camera_from_fisheye(camera)
        else:
            camera_new = perspective_camera_from_perspective(camera)
        camera_new = scale_camera(camera_new, max_size)
        camera_new.id = camera.id + "_undistorted"
        undistorter = CameraUndistorter(camera, camera_new)
    else:
        raise NotImplementedError(camera.projection_type)

    shots_undist, images_undist = undistort_shot(
        image_orig, shot, undistorter, pano_offset
    )
    for shot, image in zip(shots_undist, images_undist):
        cv2.imwrite(str(output_dir / f"{shot.id}.jpg"), image)

    return shots_undist


def pack_shot_dict(shot: Shot, info: dict) -> dict:
    latlong = info["computed_geometry"]["coordinates"][::-1]
    latlong_gps = info["geometry"]["coordinates"][::-1]
    w_p_c = shot.pose.get_origin()
    w_r_c = shot.pose.get_R_cam_to_world()
    rpy = decompose_rotmat(w_r_c)
    return dict(
        camera_id=shot.camera.id,
        latlong=latlong,
        t_c2w=w_p_c,
        R_c2w=w_r_c,
        roll_pitch_yaw=rpy,
        capture_time=info["captured_at"],
        gps_position=np.r_[latlong_gps, info["altitude"]],
        compass_angle=info["compass_angle"],
        chunk_id=int(info["sfm_cluster"]["id"]),
    )


def pack_camera_dict(camera: Camera) -> dict:
    assert camera.projection_type == "perspective"
    K = camera.get_K_in_pixel_coordinates(camera.width, camera.height)
    return dict(
        id=camera.id,
        model="PINHOLE",
        width=camera.width,
        height=camera.height,
        params=K[[0, 1, 0, 1], [0, 1, 2, 2]],
    )


def process_sequence(
    image_ids: List[int],
    image_infos: dict,
    projection: Projection,
    cfg: DictConfig,
    raw_image_dir: Path,
    out_image_dir: Path,
):
    shots = []
    image_ids = sorted(image_ids, key=lambda i: image_infos[i]["captured_at"])
    for i in image_ids:
        _, shot = opensfm_shot_from_info(image_infos[i], projection)
        shots.append(shot)
    if not shots:
        return {}
    #print(f"image_ids:{image_ids}")
    shot_idxs = keyframe_selection(shots, min_dist=cfg.min_dist_between_keyframes)
    shots = [shots[i] for i in shot_idxs]

    shots_out = thread_map(
        lambda shot: process_shot(
            shot,
            image_infos[int(shot.id)],
            raw_image_dir / image_filename.format(image_id=shot.id),
            out_image_dir,
            cfg,
        ),
        shots,
        disable=True,
    )
    shots_out = [(i, s) for i, ss in enumerate(shots_out) if ss is not None for s in ss]

    dump = {}
    for index, shot in shots_out:
        i, suffix = shot.id.rsplit("_", 1)
        info = image_infos[int(i)]
        seq_id = info["sequence"]
        is_pano = not suffix.endswith("undistorted")
        if is_pano:
            seq_id += f"_{suffix}"
        if seq_id not in dump:
            dump[seq_id] = dict(views={}, cameras={})

        view = pack_shot_dict(shot, info)
        view["index"] = index
        dump[seq_id]["views"][shot.id] = view
        dump[seq_id]["cameras"][shot.camera.id] = pack_camera_dict(shot.camera)
    return dump

def process_location(
    location: str,  # 地点名称，比如 "MGL"
    data_dir: Path,  # 存放数据的根目录路径
    split_path: Path,  # 训练/验证/测试分割的 JSON 文件路径
    token: str,  # Mapillary API 访问令牌
    cfg: DictConfig,  # 配置对象，包含各种参数设置
    generate_tiles: bool = False,  # 是否生成地图瓦片，默认为 False
):
    print(f"process_location:{location},{data_dir},{split_path},{token},{cfg},{generate_tiles}")
    params = location_to_params[location] # 从预定义参数字典中读取该地点的相关参数
    bbox = params["bbox"]  # 该地点的地理边界框（Bounding Box）
    projection = Projection(*bbox.center)  # 构建投影对象，以边界框中心为投影参考点

    splits = json.loads(split_path.read_text())  # 读取并解析分割文件，获取所有数据划分信息
    image_ids = [i for split in splits.values() for i in split[location]]  # 收集所有划分里该地点的所有图像ID
    # print(f"image_ids:{image_ids}")
    
    loc_dir = data_dir / location  # 该地点的数据目录路径
    infos_dir = loc_dir / "image_infos"  # 用于存放图像元数据的目录
    raw_image_dir = loc_dir / "images_raw"  # 用于存放原始下载图像的目录
    out_image_dir = loc_dir / "images"  # 用于存放处理后的图像的目录
    print(f"目录:{loc_dir},{infos_dir},{raw_image_dir},{out_image_dir}")
    
    for d in (infos_dir, raw_image_dir, out_image_dir):  # 遍历三个目录
        d.mkdir(parents=True, exist_ok=True)  # 若目录不存在则递归创建

    downloader = MapillaryDownloader(token)  # 创建 Mapillary 下载器对象，使用访问令牌
    print(f"我准备好了downloader:{downloader}")
    loop = asyncio.get_event_loop()  # 获取当前的异步事件循环

    logger.info("Fetching metadata for all images.")
    image_infos, num_fail = loop.run_until_complete(
        fetch_image_infos(image_ids, downloader, infos_dir)
    )
    try:
        failure_percentage = 100 * num_fail / len(image_ids) + 10
        logger.info("%d failures (%.1f%%).", num_fail, failure_percentage)
    except ZeroDivisionError:
        logger.error("No image URLs provided, cannot calculate failure percentage.")
    except Exception as e:
        logger.error("Failed to log failure information: %s", str(e))

    logger.info("Fetching image pixels.")
    image_urls = [(i, info["thumb_2048_url"]) for i, info in image_infos.items()]
    num_fail = loop.run_until_complete(
        fetch_images_pixels(image_urls, downloader, raw_image_dir)
    )
    #print(f"image_urls:{image_urls}:::{num_fail}")
    try:
        failure_percentage = 100 * num_fail / len(image_urls) + 10
        logger.info("%d failures (%.1f%%).", num_fail, failure_percentage)
    except ZeroDivisionError:
        logger.error("No image URLs provided, cannot calculate failure percentage.")
    except Exception as e:
        logger.error("Failed to log failure information: %s", str(e))

    seq_to_image_ids = defaultdict(list)
    for i, info in image_infos.items():
        seq_to_image_ids[info["sequence"]].append(i)
    seq_to_image_ids = dict(seq_to_image_ids)
    #print(f"seq_to_image_ids:{seq_to_image_ids}")
    dump = {}
    dump_path = loc_dir / "dump.json"
    # 如果 dump.json 已存在，就加载已处理的序列
    if dump_path.exists():
        dump = json.loads(dump_path.read_text())
        logger.info("Loaded existing dump with %d sequences.", len(dump))
    else:
        for seq_image_ids in tqdm(seq_to_image_ids.values()):
            dump.update(
                process_sequence(
                    seq_image_ids,
                    image_infos,
                    projection,
                    cfg,
                    raw_image_dir,
                    out_image_dir,
                )
            )
        write_json(loc_dir / "dump.json", dump)
    write_json(loc_dir / "dump1.json", dump)
    #print(f"dump:{dump}")
    # Get the view locations
    view_ids = []
    views_latlon = []
    for seq in dump:
        for view_id, view in dump[seq]["views"].items():
            view_ids.append(view_id)
            views_latlon.append(view["latlong"])
    views_latlon = np.stack(views_latlon)
    view_ids = np.array(view_ids)
    views_xy = projection.project(views_latlon)

    tiles_path = loc_dir / MapillaryDataModule.default_cfg["tiles_filename"]
    if generate_tiles:
        logger.info("Creating the map tiles.")
        bbox_data = BoundaryBox(views_xy.min(0), views_xy.max(0))
        bbox_tiling = bbox_data + cfg.tiling.margin
        osm_dir = data_dir / "osm"
        osm_path = osm_dir / params["osm_file"]
        if not osm_path.exists():
            logger.info("Downloading OSM raw data.")
            download_file(DATA_URL + f"/osm/{params['osm_file']}", osm_path)
        if not osm_path.exists():
            raise FileNotFoundError(f"Cannot find OSM data file {osm_path}.")
        tile_manager = TileManager.from_bbox(
            projection,
            bbox_tiling,
            cfg.tiling.ppm,
            tile_size=cfg.tiling.tile_size,
            path=osm_path,
        )
        tile_manager.save(tiles_path)
    else:
        tile_file = tiles_path
        if tile_file.exists() and tile_file.stat().st_size > 0:
            logger.info("Found existing tile file: %s, skip downloading.", tile_file)
        else:
            logger.info("Downloading pre-generated map tiles.")
            download_file(DATA_URL + f"/tiles/{location}.pkl", tiles_path)
        tile_manager = TileManager.load(tiles_path)

    # Visualize the data split
    plotter = GeoPlotter()
    view_ids_val = set(splits["val"][location])
    is_val = np.array([int(i.rsplit("_", 1)[0]) in view_ids_val for i in view_ids])
    plotter.points(views_latlon[~is_val], "red", view_ids[~is_val], "train")
    plotter.points(views_latlon[is_val], "green", view_ids[is_val], "val")
    plotter.bbox(bbox, "blue", "query bounding box")
    plotter.bbox(
        projection.unproject(tile_manager.bbox), "black", "tiling bounding box"
    )
    geo_viz_path = loc_dir / f"split_{location}.html"
    plotter.fig.write_html(geo_viz_path)
    logger.info("Wrote split visualization to %s.", geo_viz_path)

    shutil.rmtree(raw_image_dir)
    logger.info("Done processing for location %s.", location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument(
        "--locations", type=str, nargs="+", default=list(location_to_params)
    )  # 支持多个地点名称参数，默认处理所有地点
    parser.add_argument("--split_filename", type=str, default="splits_MGL_13loc.json")  # 分割文件名，默认值
    parser.add_argument("--token", type=str, required=True)  # Mapillary API 令牌，必填参数
    parser.add_argument(
        "--data_dir", type=Path, default=MapillaryDataModule.default_cfg["data_dir"]
    )  # 数据根目录，默认值来自配置
    parser.add_argument("--generate_tiles", action="store_true")  # 是否生成瓦片，开关型参数
    parser.add_argument("dotlist", nargs="*")  # 额外的配置参数列表（可选）
    args = parser.parse_args()  # 解析命令行参数
    print(f"{args}")
    args.data_dir.mkdir(exist_ok=True, parents=True)  # 创建数据目录（递归创建，若存在不报错）
    shutil.copy(Path(__file__).parent / args.split_filename, args.data_dir)  # 复制分割文件到数据目录
    cfg_ = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.dotlist))  # 合并默认配置和命令行传入配置
    print(f"cfg_:{cfg_}")
    i=1
    for location in args.locations:  #
        logger.info("Starting processing for location %d.", i)  # 记录日志：开始处理某地点
        logger.info("Starting processing for location %s.", location)  # 记录日志：开始处理某地点
        process_location(
            location,
            args.data_dir,
            args.data_dir / args.split_filename,
            args.token,
            cfg_,
            args.generate_tiles,
        )  # 调用处理函数进行数据准备和处理
        i=i+1