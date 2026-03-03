<div align="center">
  <h1>📍 PoiLoc</h1>
  <h3>Semantic Point-of-Interest Constrained Visual Localization for Autonomous Driving</h3>
</div>

![](https://github.com/CAVTEST/PoiLoc/blob/master/asset/method.jpg)

---

This repository provides the official implementation of a visual localization method based on point-of-interest (POI) constraints. By incorporating POI landmarks as explicit geometric constraints, PoiLoc proposes a solution to alleviate the common multimodal ambiguity problem in visually repetitive urban environments, converting indistinguishable candidate poses into reliable and highly accurate estimates.

## 🗄️ Dataset Preparation

We evaluate our method on the Mapillary Geo-Localization (MGL) dataset. For detailed instructions on downloading the dataset and generating the required OpenStreetMap (OSM) raster tiles, please visit the [OrienterNet repository](https://github.com/facebookresearch/OrienterNet).

## 🚀 Quick Start

### 1. Object Detection Model (POI Training)
To achieve robust POI recognition, the object detection model requires diverse training data. We augmented our training set using street-level images acquired from **Google Maps (Google Street View)**, capturing various urban blocks, storefronts, and functional landmarks. 

For the semantic POI recognition module, you can train your network using existing **YOLO** or **DETR** architectures on similar street-view data:
* We provide a pre-trained YOLO model for immediate testing and deployment: `[Insert Link to YOLO pre-trained weights here]`
* If you wish to construct your own dataset and train an object detection model from scratch, acquire street-level imagery from your target cities and refer to the standard training pipelines provided in the official [YOLO](https://github.com/ultralytics/yolov5) or [DETR](https://github.com/facebookresearch/detr) repositories.

### 2. Running the Localization Evaluation
The complete implementation of the POI-constrained localization pipeline, including explicit field-of-view modeling, geometric correspondence, and distance consistency verification, is contained within `poi_evaluation.py`. 

#### Evaluating Specific Cities
To test the POI localization metrics for specific urban environments, you must configure the evaluation split. Open `mapillary.py` and modify the `val` city parameter to your target city (e.g., Avignon, Toulouse, San Francisco, or Lemans).

Once the target city is set in `mapillary.py`, execute the following command to run the localization framework:

```bash
python -m model.poi_evaluation --experiment OrienterNet_MGL
```

### 3. Dataset

We evaluate our method on the Mapillary Geo-Localization (MGL) dataset. For detailed instructions on downloading the dataset and generating the required OpenStreetMap (OSM) raster tiles, please visit the [OrienterNet repository](https://github.com/facebookresearch/OrienterNet).

The trained POI detection model and localization checkpoints can be downloaded from Zenodo:

🔗 https://doi.org/10.5281/zenodo.18833037

## 🙏 Acknowledgements

This project builds upon the foundational map matching concepts and dataset preparation scripts established by OrienterNet. We sincerely thank the authors of the [OrienterNet](https://github.com/facebookresearch/OrienterNet) paper for open-sourcing their code and advancing the field of visual localization.

