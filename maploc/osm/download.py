# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional

import urllib3
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"
OSM_URL_TEMPLATE = "https://api.openstreetmap.org/api/0.6/map?bbox={left},{bottom},{right},{top}"

# def get_osm(
#     boundary_box: BoundaryBox,
#     cache_path: Optional[Path] = None,
#     overwrite: bool = False,
# ) -> Dict[str, Any]:
#     if not overwrite and cache_path is not None and cache_path.is_file():
#         return json.loads(cache_path.read_text())

#     (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
#     query = {"bbox": f"{left},{bottom},{right},{top}"}

#     logger.info("Calling the OpenStreetMap API...")
#     result = urllib3.request("GET", OSM_URL, fields=query, timeout=10)
#     if result.status != 200:
#         error = result.info()["error"]
#         raise ValueError(f"{result.status} {responses[result.status]}: {error}")

#     if cache_path is not None:
#         cache_path.write_bytes(result.data)
#     return result.json()

def get_osm(
    boundary_box: BoundaryBox,
    save_dir: Optional[Path] = None,
    overwrite: bool = False,
    proxy_url: Optional[str] = None,  # 新增参数：代理URL，默认 None
    retries: int = 3,                 # 新增参数：重试次数
    timeout: float = 10.0,            # 新增参数：超时时间
) -> str:
    if save_dir is None:
        save_dir = Path("openstreetmap")
    # 创建保存目录（如果不存在）
    save_dir.mkdir(parents=True, exist_ok=True)

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_

    # 用 bbox 信息生成唯一文件名
    filename = f"map.osm"
    file_path = save_dir / filename

    if not overwrite and file_path.is_file():
        logger.info(f"Using cached OSM file: {file_path}")
        return str(file_path.resolve())

    url = OSM_URL_TEMPLATE.format(left=left, bottom=bottom, right=right, top=top)
    logger.info(f"Downloading OSM data from: {url}")

    # 代理设置
    if proxy_url:
        http = urllib3.ProxyManager(proxy_url, headers={"User-Agent": "Mozilla/5.0"})
    else:
        http = urllib3.PoolManager(proxy_url, headers={"User-Agent": "Mozilla/5.0"})

    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            response = http.request("GET", url, timeout=timeout)
            if response.status != 200:
                error_msg = response.data.decode("utf-8")
                raise RuntimeError(f"HTTP {response.status} error: {error_msg}")
            # 成功则跳出重试循环
            break
        except (ConnectTimeoutError, MaxRetryError, RuntimeError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            last_exception = e
    else:
        # 重试结束仍失败，抛出最后异常
        raise last_exception

    # 保存响应内容到文件（OSM 是XML格式）
    file_path.write_bytes(response.data)
    logger.info(f"Saved OSM data to: {file_path}")
    # print(f"file_path.resolve(): {file_path.resolve()}")
    return str(file_path.resolve())