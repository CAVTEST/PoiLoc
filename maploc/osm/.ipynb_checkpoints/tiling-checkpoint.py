# Copyright (c) Meta Platforms, Inc. and affiliates.
# 导入标准库模块
import io  # 用于处理字节流，例如保存图像为内存文件
import pickle  # 用于对象序列化（存储）与反序列化（加载）
from pathlib import Path  # 提供路径处理的跨平台工具
from typing import Dict, List, Optional, Tuple  # 类型提示工具

# 导入第三方库
import numpy as np  # 用于数组运算和数值处理
import rtree  # 用于构建空间索引结构，支持高效的空间查询
from PIL import Image  # 图像处理库，支持读写 PNG/JPEG 等格式

# 导入项目内部模块
from ..utils.geo import BoundaryBox, Projection  # 地理边界与投影处理
from .data import MapData  # 地图数据结构定义
from .download import get_osm  # 在线获取 OSM 地图数据
from .parser import Groups  # OSM 元素分组定义
from .raster import Canvas, render_raster_map, render_raster_masks  # 栅格图像渲染函数
from .reader import OSMData, OSMNode, OSMWay  # OSM 数据结构定义



# 地图空间索引类，用于对地图节点、线段、区域构建高效索引
class MapIndex:
    def __init__(self, data: MapData):
        self.index_nodes = rtree.index.Index()  # 初始化节点索引
        for i, node in data.nodes.items():  # 遍历所有节点
            self.index_nodes.insert(i, tuple(node.xy) * 2)  # 使用节点坐标创建包围盒并插入索引

        self.index_lines = rtree.index.Index()  # 初始化线段索引
        for i, line in data.lines.items():
            bbox = tuple(np.r_[line.xy.min(0), line.xy.max(0)])  # 计算线段最小包围盒
            self.index_lines.insert(i, bbox)  # 插入索引

        self.index_areas = rtree.index.Index()  # 初始化区域索引
        for i, area in data.areas.items():
            xy = np.concatenate(area.outers + area.inners)  # 将区域的外环和内环点合并
            bbox = tuple(np.r_[xy.min(0), xy.max(0)])  # 计算最小包围盒
            self.index_areas.insert(i, bbox)  # 插入索引

        self.data = data  # 保存原始地图数据引用

    # 查询与给定边界框相交的所有节点、线段和区域
    def query(self, bbox: BoundaryBox) -> Tuple[List[OSMNode], List[OSMWay]]:
        query = tuple(np.r_[bbox.min_, bbox.max_])  # 转换边界框为平铺坐标格式
        ret = []
        for x in ["nodes", "lines", "areas"]:  # 分别查询三类元素
            ids = getattr(self, "index_" + x).intersection(query)  # 查询对应索引
            ret.append([getattr(self.data, x)[i] for i in ids])  # 返回实际对象
        return tuple(ret)


# 将边界框转换为图像中的坐标切片（用于图像裁剪）
def bbox_to_slice(bbox: BoundaryBox, canvas: Canvas):
    uv_min = np.ceil(canvas.to_uv(bbox.min_)).astype(int)  # 转换为图像坐标（像素），并上取整
    uv_max = np.ceil(canvas.to_uv(bbox.max_)).astype(int)  # 同上
    slice_ = (slice(uv_max[1], uv_min[1]), slice(uv_min[0], uv_max[0]))  # 构造切片
    return slice_


# 对边界框进行像素对齐（对齐到 ppm 定义的网格上）
def round_bbox(bbox: BoundaryBox, origin: np.ndarray, ppm: int):
    bbox = bbox.translate(-origin)  # 将边界框平移到以原点为中心
    bbox = BoundaryBox(np.round(bbox.min_ * ppm) / ppm, np.round(bbox.max_ * ppm) / ppm)  # 四舍五入对齐
    return bbox.translate(origin)  # 平移回原坐标系

# 瓦片管理类：负责地图瓦片的生成、查询、保存与加载
class TileManager:
    def __init__(
        self,
        tiles: Dict,  # 存储所有瓦片数据
        bbox: BoundaryBox,  # 整个地图的边界框
        tile_size: int,  # 单个瓦片的边长（单位：米）
        ppm: int,  # 每米像素数（pixels per meter）
        projection: Projection,  # 投影方式（经纬度 <-> XY 坐标）
        groups: Dict[str, List[str]],  # 地图元素分组信息
        map_data: Optional[MapData] = None,  # 原始地图数据
    ):
        self.origin = bbox.min_  # 原点设置为地图边界的最小坐标
        self.bbox = bbox  # 存储地图整体边界框
        self.tiles = tiles  # 存储瓦片
        self.tile_size = tile_size  # 设置瓦片边长
        self.ppm = ppm  # 设置图像分辨率
        self.projection = projection  # 存储投影信息
        self.groups = groups  # 存储地图元素组
        self.map_data = map_data  # 原始地图数据（可选）

        assert np.all(tiles[0, 0].bbox.min_ == self.origin)  # 检查左上角瓦片是否对齐地图原点
        for tile in tiles.values():
            assert bbox.contains(tile.bbox)  # 瓦片边界框必须在整个地图范围内

    # 类方法：从边界框生成瓦片地图
    @classmethod
    def from_bbox(
        cls,
        projection: Projection,
        bbox: BoundaryBox,
        ppm: int,
        path: Optional[Path] = None,
        tile_size: int = 128,
    ):
        print(f"bbox+10: {bbox}")
        bbox_osm = projection.unproject(bbox)  # 将边界框从 XY 坐标转换为经纬度
        print(f"bbox_osm: {bbox_osm}")
        if path is not None and path.is_file():
            print(f"path is not None: {path}")
            osm = OSMData.from_file(path)  # 从本地加载 OSM 数据
            # if osm.box is not None:
            #     assert osm.box.contains(bbox_osm)  # OSM 数据应包含整个地图范围
        else:
            print(f"path is  None: {path}")
            #osm = OSMData.from_dict(get_osm(bbox_osm, path,True,proxy_url="http://127.0.0.1:10809"))  # 在线下载 OSM 数据
            osm = OSMData.from_file(Path(get_osm(bbox_osm, path,True,proxy_url="http://127.0.0.1:7890")))  # 在线下载 OSM 数据
        print(f"osm Success read: {path}")
        osm.add_xy_to_nodes(projection)  # 使用投影为所有节点添加 XY 坐标
        map_data = MapData.from_osm(osm)  # 生成内部格式 MapData
        map_index = MapIndex(map_data)  # 创建空间索引

        # 构造瓦片边界框集合
        bounds_x, bounds_y = [
            np.r_[np.arange(min_, max_, tile_size), max_]  # 构造 x/y 方向上的瓦片边界线
            for min_, max_ in zip(bbox.min_, bbox.max_)
        ]
        bbox_tiles = {}  # 保存每个瓦片对应的边界框
        for i, xmin in enumerate(bounds_x[:-1]):
            for j, ymin in enumerate(bounds_y[:-1]):
                bbox_tiles[i, j] = BoundaryBox(
                    [xmin, ymin], [bounds_x[i + 1], bounds_y[j + 1]]
                )

        # 渲染瓦片图像
        tiles = {}
        for ij, bbox_tile in bbox_tiles.items():
            canvas = Canvas(bbox_tile, ppm)  # 创建画布
            nodes, lines, areas = map_index.query(bbox_tile)  # 查询当前瓦片所覆盖的地图元素
            masks = render_raster_masks(nodes, lines, areas, canvas)  # 渲染出掩码图层
            canvas.raster = render_raster_map(masks)  # 渲染最终彩色图层
            tiles[ij] = canvas  # 存储该瓦片的图像

        groups = {k: v for k, v in vars(Groups).items() if not k.startswith("__")}  # 提取分组信息

        return cls(tiles, bbox, tile_size, ppm, projection, groups, map_data)

    # 查询指定边界框的合成图像
    def query(self, bbox: BoundaryBox) -> Canvas:
        bbox = round_bbox(bbox, self.bbox.min_, self.ppm)  # 将边界框对齐像素网格
        canvas = Canvas(bbox, self.ppm)  # 创建新的画布
        raster = np.zeros((3, canvas.h, canvas.w), np.uint8)  # 初始化彩色图像缓存

        bbox_all = bbox & self.bbox  # 计算与地图范围的交集
        ij_min = np.floor((bbox_all.min_ - self.origin) / self.tile_size).astype(int)  # 起始瓦片坐标
        ij_max = np.ceil((bbox_all.max_ - self.origin) / self.tile_size).astype(int) - 1  # 结束瓦片坐标

        for i in range(ij_min[0], ij_max[0] + 1):  # 遍历所有相关瓦片
            for j in range(ij_min[1], ij_max[1] + 1):
                tile = self.tiles[i, j]  # 获取瓦片
                bbox_select = tile.bbox & bbox  # 计算交叉区域
                slice_query = bbox_to_slice(bbox_select, canvas)  # 在结果图像中的切片位置
                slice_tile = bbox_to_slice(bbox_select, tile)  # 在瓦片图像中的切片位置
                raster[(slice(None),) + slice_query] = tile.raster[
                    (slice(None),) + slice_tile
                ]  # 将瓦片图像复制到目标图像中

        canvas.raster = raster  # 设置结果图像
        return canvas

    # 将当前瓦片数据保存为本地文件
    def save(self, path: Path):
        dump = {
            "bbox": self.bbox.format(),  # 保存地图边界框
            "tile_size": self.tile_size,
            "ppm": self.ppm,
            "groups": self.groups,
            "tiles_bbox": {},  # 每个瓦片的边界框
            "tiles_raster": {},  # 每个瓦片的图像数据
        }
        if self.projection is not None:
            dump["ref_latlonalt"] = self.projection.latlonalt  # 保存投影参考点信息

        # 遍历所有瓦片
        for ij, canvas in self.tiles.items():
            dump["tiles_bbox"][ij] = canvas.bbox.format()  # 保存瓦片的边界框字符串
            raster_bytes = io.BytesIO()  # 创建字节流对象
            raster = Image.fromarray(canvas.raster.transpose(1, 2, 0).astype(np.uint8))  # 转换为 PIL 图像（HWC）
            raster.save(raster_bytes, format="PNG")  # 将图像保存为 PNG 到字节流
            dump["tiles_raster"][ij] = raster_bytes  # 保存图像字节流

        with open(path, "wb") as fp:
            pickle.dump(dump, fp)  # 将 dump 字典序列化为二进制文件

    # 从文件中加载瓦片数据
    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as fp:
            dump = pickle.load(fp)  # 从文件中反序列化瓦片数据

        tiles = {}
        for ij, bbox in dump["tiles_bbox"].items():
            tiles[ij] = Canvas(BoundaryBox.from_string(bbox), dump["ppm"])  # 创建对应瓦片画布
            raster = np.asarray(Image.open(dump["tiles_raster"][ij]))  # 读取 PNG 图像
            tiles[ij].raster = raster.transpose(2, 0, 1).copy()  # 转换回 CHW 格式图像

        projection = Projection(*dump["ref_latlonalt"])  # 创建投影对象
        return cls(
            tiles,
            BoundaryBox.from_string(dump["bbox"]),
            dump["tile_size"],
            dump["ppm"],
            projection,
            dump["groups"],
        )
