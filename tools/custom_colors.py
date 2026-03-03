import sys
import numpy as np
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

sys.path.append(str(Path(__file__).resolve().parent))

from maploc.osm.parser import Groups


class CustomColormap:
    custom_colors = {
        "building": (217, 208, 201),
        "parking": (238, 238, 238),
        "playground": (218, 255, 218),
        "grass": (205, 235, 176),
        "park": (200, 250, 204),
        "forest": (173, 209, 158),
        "water": (170, 211, 223),

        "fence": (128, 128, 128),
        "wall": (100, 100, 100),
        "hedge": (120, 150, 100),
        "kerb": (180, 180, 180),
        "building_outline": (150, 150, 150),
        "cycleway": (255, 200, 100),
        "path": (243, 197, 189),
        "road": (255, 255, 204),
        "tree_row": (150, 180, 120),
        "busway": (255, 230, 150),

        "void": (242, 239, 233),
    }

    @classmethod
    def get_colors_areas(cls):
        return np.stack([cls.custom_colors[k] for k in ["void"] + Groups.areas])

    @classmethod
    def get_colors_ways(cls):
        return np.stack([cls.custom_colors[k] for k in ["void"] + Groups.ways])

    @classmethod
    def apply(cls, rasters):
        colors_areas = cls.get_colors_areas()
        colors_ways = cls.get_colors_ways()

        return (
                np.where(
                    rasters[1, ..., None] > 0,
                    colors_ways[rasters[1]],
                    colors_areas[rasters[0]],
                )
                / 255.0
        )



