# Copyright (c) Meta Platforms, Inc. and affiliates.

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import tqdm
from aiolimiter import AsyncLimiter
from opensfm.pygeometry import Camera, Pose
from opensfm.pymap import Shot

from ... import logger
from ...utils.geo import Projection

semaphore = asyncio.Semaphore(20)  # number of parallel threads.
image_filename = "{image_id}.jpg"
info_filename = "{image_id}.json"

def retry(times, exceptions):  # 定义一个装饰器，实现重试机制
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:  # 重试次数未超过times
                try:
                    return await func(*args, **kwargs)  # 尝试执行异步函数
                except exceptions:  # 捕获指定异常则重试
                    attempt += 1
            return await func(*args, **kwargs)  # 最后一次执行
        return wrapper
    return decorator

class MapillaryDownloader:
    image_fields = (
        "id",
        "height",
        "width",
        "camera_parameters",
        "camera_type",
        "captured_at",
        "compass_angle",
        "geometry",
        "altitude",
        "computed_compass_angle",
        "computed_geometry",
        "computed_altitude",
        "computed_rotation",
        "thumb_2048_url",
        "thumb_original_url",
        "sequence",
        "sfm_cluster",
    )
    image_info_url = (
        "https://graph.mapillary.com/{image_id}?access_token={token}&fields={fields}"
    )
    seq_info_url = "https://graph.mapillary.com/image_ids?access_token={token}&sequence_id={seq_id}"  # noqa E501
    max_requests_per_minute = 50_000

    def __init__(self, token: str):  # 初始化函数，传入访问token
        self.token = token  # 保存token
        self.client = httpx.AsyncClient(  # 创建httpx异步客户端
            transport=httpx.AsyncHTTPTransport(retries=20,
            proxy="http://127.0.0.1:10808"),
            timeout=120.0  # 设置重试次数和超时时间
        )
        self.limiter = AsyncLimiter(self.max_requests_per_minute // 2, time_period=60)  # 限流器，限制请求频率
        
    @retry(times=5, exceptions=(httpx.RemoteProtocolError, httpx.ReadError))
    async def call_api(self, url: str):
        async with self.limiter:
            try:
                r = await self.client.get(url)
            except Exception as e:
                print(f"Error occurred while accessing URL: {url}")
                print(f"Error message: {e}")
                raise
        if not r.is_success:
            #logger.error("Error in API call: %s", r.text)
            print(f"Error downoading url: {url}")
        return r
    
    async def get_image_info(self, image_id: int):  # 获取单张图片信息
        url = self.image_info_url.format(  # 构造API URL
            image_id=image_id,
            token=self.token,
            fields=",".join(self.image_fields),  # 需要获取的字段
        )
        #print(f"url::{url}")
        r = await self.call_api(url)  # 调用API获取响应
        #print(f"r::{r.is_success}")
        if r.is_success:
            return json.loads(r.text)  # 解析并返回JSON内容
        
    async def get_sequence_info(self, seq_id: str):  # 获取序列内所有图片ID
        url = self.seq_info_url.format(seq_id=seq_id, token=self.token)  # 构造API URL
        r = await self.call_api(url)  # 调用API
        if r.is_success:
            return json.loads(r.text)  # 返回解析后的JSON

    async def download_image_pixels(self, url: str, path: Path):  # 下载图片数据并保存到path
        r = await self.call_api(url)  # 异步调用API获取图片内容
        if r.is_success:
            with open(path, "wb") as fid:  # 以二进制写入文件
                fid.write(r.content)
        return r.is_success  # 返回是否成功
    
    async def get_image_info_cached(self, image_id: int, path: Path):  # 获取图片信息，带缓存机制
        info = None
        if path.exists():  # 如果缓存文件存在，尝试读取
            try:
                content = path.read_text()
                if not content.strip():
                    raise ValueError("File is empty.")
                info = json.loads(content)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON from {path}, retrying from API: {e}")
                # 尝试重新从 API 下载
                info = await self.get_image_info(image_id)
                if info is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(info, indent=2))
        else:
            # 如果缓存文件不存在，直接调用 API 下载
            info = await self.get_image_info(image_id)
            if info is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(info, indent=2))

        return info
    
    async def download_image_pixels_cached(self, url: str, path: Path):  # 下载图片，带缓存机制
        if path.exists():  # 文件存在直接返回True
            if path.stat().st_size > 0:  # 检查文件大小是否大于0
                #print("文件已存在且有内容，无需重新下载")
                return True
            else:
                print("文件已存在没有内容，重新下载")
                return await self.download_image_pixels(url, path)  # 否则下载保存
        else:
            #print("正在下载")
            #print(f"path:{path}")
            return await self.download_image_pixels(url, path)  # 否则下载保存


async def fetch_images_in_sequence(i, downloader):  # 异步获取序列内图片ID列表
    async with semaphore:  # 控制并发数
        info = await downloader.get_sequence_info(i)  # 调用下载器获取序列信息
    if info is None:
        image_ids = None
    else:
        image_ids = [int(d["id"]) for d in info["data"]]  # 解析图片ID列表
    return i, image_ids  # 返回序列ID及图片ID列表

async def fetch_images_in_sequences(sequence_ids, downloader):  # 批量获取多个序列的图片ID列表
    seq_to_images_ids = {}
    tasks = [fetch_images_in_sequence(i, downloader) for i in sequence_ids]  # 创建任务列表
    for task in tqdm.asyncio.tqdm.as_completed(tasks):  # 迭代异步完成的任务，带进度条
        i, image_ids = await task
        if image_ids is not None:
            seq_to_images_ids[i] = image_ids  # 保存非空结果
    return seq_to_images_ids  # 返回序列到图片ID的字典

async def fetch_image_info(i, downloader, dir_):  # 异步获取单张图片信息
    async with semaphore:  # 限制并发数
        #print("async with semaphore:")
        path = dir_ / info_filename.format(image_id=i)  # 缓存文件路径
        info = await downloader.get_image_info_cached(i, path)  # 先尝试从缓存获取
        #print(f"async with semaphore")  # 打印信息调试
    return i, info  # 返回图片ID和信息

async def fetch_image_infos(image_ids, downloader, dir_):  # 批量异步获取图片信息
    #print(f"fetch_image_infos,{downloader}:")
    #print(f"fetch_image_infos,{dir_}:")
    infos = {}  # 保存结果字典
    num_fail = 0  # 失败计数
    tasks = [fetch_image_info(i, downloader, dir_) for i in image_ids]  # 任务列表
    print("no inst")
    for task in tqdm.asyncio.tqdm.as_completed(tasks):  # 异步迭代任务完成，带进度条
        i, info = await task
        if info is None:
            num_fail += 1  # 统计失败
        else:
            infos[i] = info  # 保存成功的结果
        #print(f"num_fail,{num_fail}:")
        #print(f"infos,{infos}:")
    print("no inst")
    return infos, num_fail  # 返回所有信息和失败数


async def fetch_image_pixels(i, url, downloader, dir_, overwrite=False):  # 异步下载单张图片数据
    async with semaphore:  # 控制并发数
        path = dir_ / image_filename.format(image_id=i)  # 图片保存路径
        if overwrite:
            path.unlink(missing_ok=True)  # 覆盖时删除已有文件
        success = await downloader.download_image_pixels_cached(url, path)  # 下载或使用缓存
    return i, success  # 返回图片ID和是否成功


async def fetch_images_pixels(image_urls, downloader, dir_):  # 批量异步下载图片数据
    num_fail = 0  # 失败计数
    tasks = [fetch_image_pixels(*id_url, downloader, dir_) for id_url in image_urls]  # 任务列表
    for task in tqdm.asyncio.tqdm.as_completed(tasks):  # 异步迭代完成任务，带进度条
        i, success = await task
        num_fail += not success  # 失败计数累加
    return num_fail  # 返回失败数量


def opensfm_camera_from_info(info: dict) -> Camera:
    cam_type = info["camera_type"]
    if cam_type == "perspective":
        camera = Camera.create_perspective(*info["camera_parameters"])
    elif cam_type == "fisheye":
        camera = Camera.create_fisheye(*info["camera_parameters"])
    elif Camera.is_panorama(cam_type):
        camera = Camera.create_spherical()
    else:
        raise ValueError(cam_type)
    camera.width = info["width"]
    camera.height = info["height"]
    camera.id = info["id"]
    return camera


def opensfm_shot_from_info(info: dict, projection: Projection) -> Shot:
    latlong = info["computed_geometry"]["coordinates"][::-1]
    alt = info["computed_altitude"]
    xyz = projection.project(np.array([*latlong, alt]), return_z=True)
    c_rotvec_w = np.array(info["computed_rotation"])
    pose = Pose()
    pose.set_from_cam_to_world(-c_rotvec_w, xyz)
    camera = opensfm_camera_from_info(info)
    return latlong, Shot(info["id"], camera, pose)
