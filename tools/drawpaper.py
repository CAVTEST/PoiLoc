
import sys

import cv2
from matplotlib.font_manager import FontProperties
from pathlib import Path
import maploc.data.json as json

from model.poi_localization_pipeline import POILocalizationPipeline

from tools.custom_colors import CustomColormap

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.yolo_poi_detector import YOLOPOIDetector

from maploc.utils.io import read_image
import matplotlib.pyplot as plt

from maploc.demo import Demo
from maploc.osm.viz import GeoPlotter
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap
from maploc.utils.viz_2d import plot_images
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import features_to_RGB

demo = Demo(num_rotations=256, device="cpu")

selected_image_path = json.select_image_file()
if selected_image_path:
    image_filename = Path(selected_image_path).stem
    path_parts = Path(selected_image_path).parts
    city_name = path_parts[5]
    pose_json_path = f"D:/SoftWare/Desktop/OrienterNet/my_results/{city_name}/"
    prior_latlon = json.get_pose_info_from_json1(image_filename, pose_json_path, posecls="ground_truth")
    actual_latlon = json.get_pose_info_from_json1(image_filename, pose_json_path, posecls="ground_truth")

    if prior_latlon:

        image_path = selected_image_path
        prior_latlon = tuple(prior_latlon)

        print(f"image_path = \"{image_path}\"")
        print(f"prior_latlon = {prior_latlon}")

        image, camera, gravity, proj, bbox = demo.read_input_image(
            image_path,
            prior_latlon=prior_latlon,
            tile_size_meters=64,  # try 64, 256, etc.
        )
        print(f"bbox{bbox}")
        # Show the query area in an interactive map
        plot = GeoPlotter(zoom=16)
        plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
        plot.bbox(proj.unproject(bbox), "blue", name="map tile")
        plot.fig.show()

        # Query OpenStreetMap for this area
        tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
        canvas = tiler.query(bbox)

        osm_image, total_path = json.show_osm_image(selected_image_path, image_filename)
        # Show the inputs to the model: image and raster map
        map_viz = Colormap.apply(canvas.raster)
        map_yoloviz = CustomColormap.apply(canvas.raster)
        uv, yaw, prob, neural_map, image_rectified = demo.localize(
            image, camera, canvas, gravity=gravity
        )
        overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
        (neural_map_rgb,) = features_to_RGB(neural_map.numpy())

        pred_latlon = proj.unproject(canvas.to_xy(uv))
        print(f"(lat, lon): ({float(pred_latlon[0]):.8f}, {float(pred_latlon[1]):.8f})")
        distance = json.haversine_distance(pred_latlon, actual_latlon)

        imageyolo = cv2.imread(image_path)
        yolo = YOLOPOIDetector(f"D:/SoftWare/Desktop/PaperCode/MGL/{city_name}/yolo.pt")
        yolo_detections = yolo.detect_pois(imageyolo)

        yolo_vis = yolo.visualize_detections(imageyolo, yolo_detections)
        yolo_output = f"yolo.jpg"
        cv2.imwrite(str(yolo_output), yolo_vis)
        image_yolo = read_image(yolo_output)

        pipeline = POILocalizationPipeline(f"D:/SoftWare/Desktop/PaperCode/MGL/{city_name}/yolo.pt")

        output_dir = Path("poi_localization_results")
        output_dir.mkdir(exist_ok=True)

        result = pipeline.localize_from_image(
            image_path,
            prior_latlon=prior_latlon,
            tile_size_meters=64,
            output_dir=output_dir,
            actual_latlon=actual_latlon
        )
        poi_image = f"poi_image.jpg"
        poi_yolo = read_image(poi_image)

        poi_recoogion = f"poi_image_poi.jpg"
        map_yoloviz = read_image(poi_recoogion)

        plot_images([image, neural_map_rgb, overlay, map_yoloviz, poi_yolo, image],
                    titles=["input image", "neural map", "prediction position", "poi recognition", "final position",
                            "image_yolo"])
        # plot_images([image,neural_map_rgb,overlay,image_yolo,poi_yolo])
        ax = plt.gcf().axes[2]
        ax1 = plt.gcf().axes[0]
        ax2 = plt.gcf().axes[3]
        ax3 = plt.gcf().axes[5]
        ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
        plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
        add_circle_inset(ax, uv)

        if yolo_detections and len(yolo_detections) > 0:
            yolo_center = yolo_detections[0]['center']  # [cx, cy]
            yolo_uv = yolo_center
            add_circle_inset(ax3, yolo_uv, ax2, radius_px=50)
        else:
            add_circle_inset(ax1, uv, ax2, radius_px=20)

        ax3.axis('off')

        font_path = None
        font_prop = FontProperties(size=28)

        distance_text = f"ΔGPS={distance:.2f} m"
        ax.text(
            0.02,
            0.98,
            distance_text,
            transform=ax.transAxes,
            fontproperties=font_prop,
            color="yellow",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
        )
        # plt.tight_layout()

        fig = plt.gcf()
        fig.delaxes(ax3)

        overlay_path = f"{image_filename}_paper.png"
        plt.savefig(overlay_path, dpi=1200, bbox_inches='tight')




