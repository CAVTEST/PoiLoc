# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.nn.functional as F

from maploc.demo import ImageCalibrator
from maploc.evaluation.mapillary import run as run_mapillary_evaluation
from maploc.osm.reader import OSMData
from maploc.utils.exif import EXIF
from maploc.utils.io import read_image
from model.raw_osm_processor import RawOSMProcessor
from model.yolo_poi_detector import YOLOPOIDetector

evaluation_results = []


def single_image_callback(i, model, pred, batch, results):
    """
    This callback is executed for each single image during evaluation.
    """
    global evaluation_results

    # Extract the required information from the batch and prediction
    image_path = batch.get('path')
    camera = batch.get('camera')
    proj = batch.get('proj')
    tiler = batch.get('tiler')
    canvas = batch.get('canvas')
    bbox = batch.get('bbox')
    log_probs = pred.get('log_probs')

    # Extract initial and ground truth positions
    uv_init = batch.get('uv_init')  # Initial UV coordinates
    uv_gt = batch.get('uv')  # Ground truth UV coordinates
    xy_w_gt = batch.get('xy_w_gt')
    latlon_gt = proj.unproject(xy_w_gt.reshape(1, -1))[0]
    yaw_gt = batch.get('yaw_gt')

    uv_max = pred.get('uv_max')  # Maximum likelihood position
    xy_max = canvas.to_xy(uv_max.cpu().numpy() if hasattr(uv_max, 'cpu') else uv_max)
    latlon_max = proj.unproject(xy_max.reshape(1, -1))[0]
    yaw_max = pred.get('yaw_max')  # Maximum likelihood yaw

    uv_expectation = pred.get('uv_expectation')  # Expected position

    # Extract additional useful data
    scene = batch.get('scene')
    sequence = batch.get('sequence')
    name = batch.get('name')
    roll_pitch_yaw = batch.get('roll_pitch_yaw')

    print(f"--- Image Index: {i} ---")
    print(f"Image Path: {image_path}")

    print("Step 1: Loading and preprocessing image...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(image_path)

    print("Step 2: Detecting POIs with YOLO...")
    path_parts = image_path.split(os.sep)
    city_name = path_parts[5]
    yolo_detections = YOLOPOIDetector(f"D:/SoftWare/Desktop/PaperCode/MGL/{city_name}/yolo.pt").detect_pois(image)
    if not yolo_detections:
        print("No POIs detected in the image. Stopping pipeline.")
        image_result = {
            'image_index': i,
            'image_path': str(image_path),
            'scene': scene,
            'sequence': sequence,
            'name': name,
            'uv_max': uv_max,
            'xy_max': xy_max,
            'latlon_max': latlon_max,
            'uv_gt': uv_gt,
            'xy_w_gt': xy_w_gt,
            'latlon_gt': latlon_gt,
            'yaw_gt': yaw_gt,
            'status': 'success',
            'xy_pred': xy_max,
            'yaw_pred': yaw_max,
            'yaw_max': yaw_max
        }
        evaluation_results.append(image_result)
        xy_error_computed = calculate_xy_error(xy_max, xy_w_gt)
        yaw_error_computed = calculate_yaw_error(yaw_max, yaw_gt)
        poixy_error_computed = calculate_xy_error(xy_max, xy_w_gt)
        poiyaw_error_computed = calculate_yaw_error(yaw_max, yaw_gt)
        print(f"\033[31m error_computed:({xy_error_computed, poixy_error_computed})\033[0m")
        print(f"\033[31m yaw_error_computed:({yaw_error_computed, poiyaw_error_computed})\033[0m")
    else:
        distance = math.sqrt((xy_max[0] - xy_w_gt[0]) ** 2 + (xy_max[1] - xy_w_gt[1]) ** 2)
        print(f"\033[31m {image_path},Distance: {distance:.10f}\033[0m")

        image = read_image(image_path)
        with open(image_path, "rb") as fid:
            exif = EXIF(fid, lambda: image.shape[:2])
        focal_length = None
        gravity, camera = ImageCalibrator().to(device).run(image, focal_length, exif)
        focal_length = camera.data[2]
        image_width = camera.data[0]
        image_height = camera.data[1]
        print(f"Got probability distribution with focal_length: {focal_length}")

        print("Step 4: Loading OSM data...")
        try:
            osm_dir = "D:/SoftWare/Desktop/OrienterNet/openstreetmap"
            image_name = Path(image_path).stem
            osm_path = Path(osm_dir) / f"{city_name}" / image_name / "map.osm"
            print(f"osm_path:{osm_path}")
            if osm_path is not None and osm_path.is_file():
                osm = OSMData.from_file(osm_path)
                osm.add_xy_to_nodes(proj)
            else:
                print(f"waring！")
            osm_pois = RawOSMProcessor().extract_pois_from_osm(osm, proj, bbox)

        except Exception as e:
            print(f"Error loading OSM data: {e}")
        print(f" uv_max: ({uv_gt[0]:.10f}, {uv_gt[1]:.10f}),({xy_w_gt[0]:.10f}, {xy_w_gt[1]:.10f})")
        print(f" xy_max: ({uv_max[0]:.10f}, {uv_max[1]:.10f}),({xy_max[0]:.10f}, {xy_max[1]:.10f})")
        print("Step 5: Field-of-view based POI verification...")

        xy_pred = None
        yaw_pred = None

        if log_probs is not None and canvas is not None and proj is not None:
            candidate_locations = extract_candidates_from_log_probs_original(log_probs, canvas, proj, xy_w_gt,
                                                                             thresh=0.0001)

            if candidate_locations:
                print(f"Verifying {len(candidate_locations)} location candidates...")
                verification_results = []
                for i, candidate in enumerate(candidate_locations):
                    print(f"  Verifying candidate {i + 1}/{len(candidate_locations)}...")

                    result = _verify_pois_in_field_of_view(yolo_detections, candidate, osm_pois,
                                                           image_width=image_width, image_height=image_height,
                                                           focal_length=focal_length, actual_yaw=yaw_gt)
                    verification_results.append(result)

                best_result = max(verification_results, key=lambda x: x['verification_score'])

                matches = []
                for verified_match in best_result['verified_matches']:
                    matches.append({
                        'yolo_detection': verified_match['yolo_detection'],
                        'osm_poi': verified_match['matched_poi'],
                        'match_type': verified_match['match_type'],
                        'confidence': verified_match['match_score'],
                        'distance': verified_match['actual_distance'],
                        'estimated_distance': verified_match['estimated_distance'],
                        'position_error': verified_match['position_error'],
                        'angle_diff': verified_match['angle_diff'],
                        'candidate': verified_match['candidate']
                    })

                estimated_location = (best_result['candidate']['lat'], best_result['candidate']['lon'])
                best_candidate = best_result['candidate']

                print(f"Best candidate: score={best_result['verification_score']:.3f}, "
                      f"verified_matches={best_result['num_verified']}, "
                      f"location=({estimated_location[0]:.6f}, {estimated_location[1]:.6f}), "
                      f"yaw={best_candidate['yaw']:.1f}°")

                print(f"best_candidate:{best_candidate['uv']}")
                xy_pred = np.array([best_candidate['xy'][0], best_candidate['xy'][1]])
                yaw_pred = best_candidate['yaw']

            else:
                print("No valid location candidates found")

            print(f"final location world coordinates: ({xy_pred[0]:.2f}, {xy_pred[1]:.2f})")

        else:
            print("Step 5: Fallback POI matching (no probability distribution)...")

        result = {
            'status': 'success',
            'image_index': i,
            'image_path': str(image_path),
            'scene': scene,
            'sequence': sequence,
            'name': name,
            'uv_max': uv_max,
            'xy_max': xy_max,
            'latlon_max': latlon_max,
            'yaw_max': yaw_max,
            'uv_gt': uv_gt,
            'xy_w_gt': xy_w_gt,
            'latlon_gt': latlon_gt,
            'yaw_gt': yaw_gt,
            'xy_pred': xy_pred,
            'yaw_pred': yaw_pred,
        }
        evaluation_results.append(result)
        xy_error = calculate_xy_error(xy_max, xy_w_gt)
        yaw_error = calculate_yaw_error(yaw_max, yaw_gt)
        xypred_error = calculate_xy_error(xy_pred, xy_w_gt)
        yawpred_error = calculate_yaw_error(yaw_pred, yaw_gt)
        print(f"\033[31m error_computed:({xy_error, xypred_error})\033[0m")
        print(f"\033[31m yaw_error_computed:({yaw_error, yawpred_error})\033[0m")
        print("-------------------------\n")


def _verify_pois_in_field_of_view(yolo_detections, candidate, osm_pois,
                                  image_width=512, image_height=512, focal_length=None, actual_yaw=None):
    camera_xy = candidate['xy']
    camera_uv = candidate['uv']
    camera_yaw = candidate['yaw']
    distance = candidate['distance']
    print(
        f"Verifying FOV for candidate at ({candidate['lat']:.6f}, {candidate['lon']:.6f}) ,({camera_uv[0]},{camera_uv[1]}),({camera_xy[0]:.4f},{camera_xy[1]:.4f})"
        f",yaw={camera_yaw:.1f}°,actual_yaw={actual_yaw:.1f}")

    if focal_length is None:
        fov_horizontal = 60.0
        focal_length = image_width / (2 * np.tan(np.radians(fov_horizontal / 2)))

    fov_horizontal = np.rad2deg(2 * np.arctan(image_width / (2 * focal_length)))
    fov_vertical = np.rad2deg(2 * np.arctan(image_height / (2 * focal_length)))
    max_distance = 62.0

    print(f"  Camera params: focal={focal_length:.1f}px, FOV={fov_horizontal:.1f}°×{fov_vertical:.1f}°")
    verified_matches = []
    for yolo_detection in yolo_detections:
        yolo_class = yolo_detection['class_name']
        bbox = yolo_detection['bbox']  # [x1, y1, x2, y2]
        center = yolo_detection['center']  # [cx, cy]

        relative_x = (center[0] - image_width / 2) / (image_width / 2)  # [-1, 1]
        relative_y = (center[1] - image_height / 2) / (image_height / 2)  # [-1, 1]

        angle_offset_x = np.degrees(np.arctan(relative_x * image_width / (2 * focal_length)))
        angle_offset_y = np.degrees(np.arctan(relative_y * image_height / (2 * focal_length)))

        bbox_height = bbox[3] - bbox[1]
        estimated_distance = _estimate_distance_from_bbox(bbox_height, yolo_class, focal_length)

        target_yaw = camera_yaw + angle_offset_x
        target_distance = min(estimated_distance, max_distance)

        target_x = camera_xy[0] + target_distance * np.sin(np.radians(target_yaw))
        target_y = camera_xy[1] + target_distance * np.cos(np.radians(target_yaw))

        print(f"  YOLO {yolo_class}: img_pos=({relative_x:.2f},{relative_y:.2f}), "
              f"angle_offset={angle_offset_x:.1f}°, est_dist={estimated_distance:.1f}m")
        print(f"    Target map position: ({target_x:.1f}, {target_y:.1f})")

        matching_osm_pois = []

        for poi in osm_pois:
            if RawOSMProcessor()._is_category_match(yolo_class, poi.category, poi.name):
                matching_osm_pois.append(poi)

        if not matching_osm_pois:
            print(f"\033[31m    No matching OSM POIs found for {yolo_class}\033[0m")
            continue

        best_match = None
        best_score = 0

        for poi in matching_osm_pois:
            if poi.xy is None:
                print(f"\033[31m    POI {poi} does not have xy coordinates.\033[0m")
                continue

            position_error = np.sqrt((poi.xy[0] - target_x) ** 2 + (poi.xy[1] - target_y) ** 2)

            dx = poi.xy[0] - camera_xy[0]
            dy = poi.xy[1] - camera_xy[1]
            actual_distance = np.sqrt(dx ** 2 + dy ** 2)

            angle = np.degrees(np.arctan2(dy, dx))

            clockwise_angle = 90 - angle
            if clockwise_angle < 0:
                clockwise_angle += 360
            half_fov = fov_horizontal / 2
            lower_bound = (camera_yaw - half_fov) % 360
            upper_bound = (camera_yaw + half_fov) % 360
            actual_angle = clockwise_angle - camera_yaw
            if lower_bound <= upper_bound:
                if lower_bound <= clockwise_angle <= upper_bound:
                    actual_angle = clockwise_angle - camera_yaw
                else:
                    continue
            else:
                if clockwise_angle >= lower_bound or clockwise_angle <= upper_bound:
                    actual_angle = clockwise_angle - camera_yaw
                else:
                    continue
            if actual_distance > max_distance:
                continue
            distance_score = _calculate_distance_match_score(
                estimated_distance, actual_distance, position_error
            )
            angle_score = _calculate_angle_match_score(angle_offset_x, actual_angle)
            # distance_score * 0.7 +  angle_score * 0.3
            total_score = distance_score * 0.7 + angle_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_match = {
                    'poi': poi,
                    'actual_distance': actual_distance,
                    'estimated_distance': estimated_distance,
                    'position_error': position_error,
                    'angle_diff': actual_angle,
                    'angle_offset_expected': angle_offset_x,
                    'match_score': best_score,
                    'candidate': candidate,
                }

        if best_match and best_score > 0.1:
            verified_matches.append({
                'yolo_detection': yolo_detection,
                'matched_poi': best_match['poi'],
                'actual_distance': best_match['actual_distance'],
                'estimated_distance': best_match['estimated_distance'],
                'position_error': best_match['position_error'],
                'angle_diff': best_match['angle_diff'],
                'match_score': best_match['match_score'],
                'match_type': 'fov_position_verified',
                'candidate': best_match['candidate']
            })

            print(f"\033[92m    ✓ Matched with {best_match['poi'].name}: "
                  f"dist_err={abs(best_match['actual_distance'] - best_match['estimated_distance']):.1f}m, "
                  f"poipos_err={best_match['position_error']:.1f}m, score={best_match['match_score']:.3f} \033[0m")
        else:
            print(f"    ✗ No suitable match found (best_score={best_score:.3f})")

    verification_score = _calculate_verification_score(verified_matches, candidate)
    print(f"\033[92m     ✓ verification_score: {verification_score} \033[0m")
    return {
        'candidate': candidate,
        'verified_matches': verified_matches,
        'verification_score': verification_score,
        'num_verified': len(verified_matches)
    }


def _calculate_verification_score(verified_matches, candidate):
    if not verified_matches:
        location_score = candidate.get('prob_value', 0.0)
        return location_score * 0.1

    base_score = len(verified_matches) * 3.0

    quality_score = 0.0
    total_position_accuracy = 0.0
    total_distance_accuracy = 0.0

    for match in verified_matches:
        match_score = match.get('match_score', 0.0)
        quality_score += match_score

        position_error = match.get('position_error', 50.0)  # 8.7 7.0
        position_accuracy = max(0, 1.0 - position_error / 30.0)
        total_position_accuracy += position_accuracy

        estimated_dist = match.get('estimated_distance', 50.0)
        actual_dist = match.get('actual_distance', 50.0)
        distance_error_ratio = abs(estimated_dist - actual_dist) / max(estimated_dist, actual_dist)
        distance_accuracy = max(0, 1.0 - distance_error_ratio)
        total_distance_accuracy += distance_accuracy

    avg_quality = quality_score / len(verified_matches)
    avg_position_accuracy = total_position_accuracy / len(verified_matches)
    avg_distance_accuracy = total_distance_accuracy / len(verified_matches)

    prob_value = candidate.get('prob_value', 1e-10)  # prob_value
    location_score = prob_value * 5.0

    total_score = (base_score * 0.45 +
                   avg_quality * 0.35 +
                   # avg_position_accuracy * 0.15 +
                   # avg_distance_accuracy * 0.15 +
                   location_score * 0.20
                   )

    return total_score


def extract_candidates_from_log_probs_original(log_probs, canvas, proj, xy_w_gt, thresh=0.001, k=3, top_k=20):
    """
    Extracts candidate locations from a log probability volume,
    strictly following the logic from the original project's
    `plot_dense_rotations` function.
    """
    if log_probs is None:
        return []

    # The original function works with probabilities, not log_probs
    probs = torch.exp(log_probs.cpu())

    # 1. Get the best rotation index and yaw for each pixel
    rot_indices_map = torch.argmax(probs, -1)
    yaws_map = rot_indices_map.numpy() / probs.shape[-1] * 360
    # 2. Get the 2D spatial probability map and normalize it
    spatial_probs = probs.max(-1).values
    if spatial_probs.max() > 0:
        spatial_probs = spatial_probs / spatial_probs.max()

    # 3. Create mask based on threshold
    mask = spatial_probs > thresh

    # 4. Use max_pool2d to find local maxima (Non-Maximum Suppression)
    masked = spatial_probs.masked_fill(~mask, 0)
    max_pooled = F.max_pool2d(
        masked.float().unsqueeze(0).unsqueeze(0),
        kernel_size=k,
        stride=1,
        padding=k // 2
    ).squeeze(0).squeeze(0)

    # A point is a local peak if its value is equal to the max in its neighborhood
    local_maxima_mask = (max_pooled == masked.float()) & mask

    # 5. Get the coordinates of the peaks
    candidate_coords = torch.nonzero(local_maxima_mask, as_tuple=False)  # Returns [[y1, x1], [y2, x2], ...]

    # 6. Extract info for each candidate
    candidates = []
    for coords in candidate_coords:
        y, x = coords[0].item(), coords[1].item()
        # Get the score (normalized probability)
        score = spatial_probs[y, x].item()
        try:
            # Get raw probability value (not normalized)
            rot_idx = rot_indices_map[y, x].item()
            prob_raw_value = probs[y, x, rot_idx].item()
            # Get yaw angle
            yaw_angle = yaws_map[y, x]
            # Get uv and xy coordinates
            uv_pred = [x, y]
            xy_pred = canvas.to_xy(np.array(uv_pred))
            # Get lat/lon coordinates
            latlon_pred = proj.unproject(xy_pred)
            # Calculate distance from origin (0,0) in xy plane
            distance = math.sqrt((xy_pred[0] - xy_w_gt[0]) ** 2 + (xy_pred[1] - xy_w_gt[1]) ** 2)
            candidates.append({
                'lat': float(latlon_pred[0]),
                'lon': float(latlon_pred[1]),
                'yaw': float(yaw_angle),
                'prob_value': prob_raw_value,
                'uv': [float(uv_pred[0]), float(uv_pred[1])],
                'xy': [float(xy_pred[0]), float(xy_pred[1])],
                'grid_coords': [int(y), int(x)],
                'distance': distance,
                'score': score
            })
        except Exception as e:
            print(f"Warning: Could not process candidate at uv=({x},{y}): {e}")
    # 7. Sort by score and take top_k
    candidates.sort(key=lambda c: c['score'], reverse=True)

    print(f"Extracted {len(candidates)} location candidates:")
    for i, cand in enumerate(candidates):
        print(f"  {i + 1}. ({cand['lat']:.6f}, {cand['lon']:.6f}), "
              f"uv=({cand['uv'][0]}, {cand['uv'][1]}), "
              f"xy=({cand['xy'][0]:.6f}, {cand['xy'][1]:.6f}), "
              f"distance=({cand['distance']:.6f}), "
              f"yaw={cand['yaw']:.1f}° prob={cand['prob_value']:.15f}")

    return candidates[:top_k]


def _calculate_distance_match_score(estimated_distance, actual_distance, position_error):
    distance_error = abs(estimated_distance - actual_distance)
    distance_error_ratio = distance_error / max(estimated_distance, actual_distance)

    distance_score = max(0, 1.0 - distance_error_ratio)

    position_score = max(0, 1.0 - position_error / 20.0)

    return distance_score * 0.6 + position_score * 0.4


def _calculate_angle_match_score(expected_angle_offset, actual_angle_diff):
    angle_error = abs(expected_angle_offset - actual_angle_diff)

    angle_score = max(0, 1.0 - angle_error / 30.0)

    return angle_score


def _estimate_distance_from_bbox(bbox_height, object_class, focal_length):
    typical_heights = {
        'person': 1.7,
        'car': 1.5,
        'truck': 3.0,
        'bus': 3.5,
        'building': 3.5,
        'restaurant': 3.0,
        'coffee': 2.5,
        'hotel': 4.0,
        'shop': 2.5,
        'bank': 3.0,
        'default': 4.0,
        'crossing': 0,
        'lamp': 4,
        '3H Service Market': 4,
    }

    real_height = typical_heights.get(object_class.lower(), typical_heights['default'])

    # distance = (real_height * focal_length) / bbox_height
    if bbox_height > 0:
        estimated_distance = (real_height * focal_length) / bbox_height
        estimated_distance = max(0, min(estimated_distance, 64.0))
    else:
        estimated_distance = 15.0

    print(f"estimated distance is {estimated_distance},real_height:{real_height}")

    return estimated_distance


def calculate_overall_recall(results):
    """
    Calculate overall recall rates and save results to JSON file.
    """
    print(f"Total images processed: {len(results)}")

    thresholds = [1, 2, 3, 5, 10]

    position_errors = []
    positionmax_errors = []
    positionX_errors = []
    positionY_errors = []
    angle_errors = []
    anglemax_errors = []

    for result in results:
        dx = result['xy_pred'][0] - result['xy_w_gt'][0]
        dy = result['xy_pred'][1] - result['xy_w_gt'][1]
        dxmax = result['xy_max'][0] - result['xy_w_gt'][0]
        dymax = result['xy_max'][1] - result['xy_w_gt'][1]

        position_error = np.sqrt(dx ** 2 + dy ** 2)
        positionmax_error = np.sqrt(dxmax ** 2 + dymax ** 2)
        position_errors.append(position_error)
        positionmax_errors.append(positionmax_error)
        positionX_errors.append(abs(dx))
        positionY_errors.append(abs(dy))

        angle_diff = abs(result['yaw_pred'] - result['yaw_gt'])
        angle_error = min(angle_diff, 360 - angle_diff)
        angle_errors.append(angle_error)

        angle_diff = abs(result['yaw_max'] - result['yaw_gt'])
        angle_error = min(angle_diff, 360 - angle_diff)
        anglemax_errors.append(angle_error)

    print(f"poiXY:")
    xy_recall = compute_recall_at_thresholds(position_errors, thresholds)
    for i, threshold in enumerate(thresholds):
        count = np.sum(np.array(position_errors) <= threshold)
        print(f"  <= {threshold}m: {count}/{len(position_errors)} ({xy_recall[i]}%)")

    print(f"Poix:")
    poix_recall = compute_recall_at_thresholds(positionX_errors, thresholds)
    for i, threshold in enumerate(thresholds):
        count = np.sum(np.array(positionX_errors) <= threshold)
        print(f"  <= {threshold}°: {count}/{len(positionX_errors)} ({poix_recall[i]}%)")

    print(f"Poiy:")
    poiy_recall = compute_recall_at_thresholds(positionY_errors, thresholds)
    for i, threshold in enumerate(thresholds):
        count = np.sum(np.array(positionY_errors) <= threshold)
        print(f"  <= {threshold}°: {count}/{len(positionY_errors)} ({poiy_recall[i]}%)")

    print(f"poiYaw:")
    yaw_recall = compute_recall_at_thresholds(angle_errors, thresholds)
    for i, threshold in enumerate(thresholds):
        count = np.sum(np.array(angle_errors) <= threshold)
        print(f"  <= {threshold}°: {count}/{len(angle_errors)} ({yaw_recall[i]}%)")

    mean_position_error = np.mean(position_errors)
    print(f"PoI Mean Position Error: {mean_position_error:.4f} meters")

    mean_angle_error = np.mean(angle_errors)
    print(f"POI Mean Angle Error: {mean_angle_error:.4f} degrees")

    meanmax_position_error = np.mean(positionmax_errors)
    print(f"Mean Position Error: {meanmax_position_error:.4f} meters")

    meanmax_angle_error = np.mean(anglemax_errors)

    # =======================
    # CSV
    # =======================
    with open("avignon.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        header = ["Metric"]
        for t in thresholds:
            header.append(f"Recall@{t}")
        writer.writerow(header)

        writer.writerow(["XY Recall"] + list(xy_recall))
        writer.writerow(["X Recall"] + list(poix_recall))
        writer.writerow(["Y Recall"] + list(poiy_recall))
        writer.writerow(["Yaw Recall"] + list(yaw_recall))

        writer.writerow([])
        writer.writerow(["Mean Position Error (PoI)", mean_position_error])
        writer.writerow(["Mean Angle Error (PoI)", mean_angle_error])
        writer.writerow(["Mean Position Error (Max)", meanmax_position_error])
        writer.writerow(["Mean Angle Error (Max)", meanmax_angle_error])

    print(f"\nEvaluation summary saved to")


def calculate_xy_error(xy_pred, xy_gt):
    xy_pred = np.array(xy_pred)
    xy_gt = np.array(xy_gt)
    xy_error = np.linalg.norm(xy_pred - xy_gt)
    return float(xy_error)


def calculate_yaw_error(yaw_pred, yaw_gt):
    yaw_pred_rad = math.radians(yaw_pred)
    yaw_gt_rad = math.radians(yaw_gt)

    diff = yaw_pred_rad - yaw_gt_rad

    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    yaw_error = abs(math.degrees(diff))
    return float(yaw_error)


def compute_recall_at_thresholds(errors, thresholds):
    errors = np.array(errors)
    total_samples = len(errors)

    recall_values = []
    for threshold in thresholds:
        correct_samples = np.sum(errors <= threshold)
        recall = (correct_samples / total_samples) * 100
        recall_values.append(round(recall, 2))

    return recall_values


def run(
        split: str,
        experiment: str,
        **kwargs,
):
    global evaluation_results
    evaluation_results = []  # Reset results

    print("Starting single-image evaluation with custom callback...")

    run_mapillary_evaluation(
        split=split,
        experiment=experiment,
        sequential=False,  # Always run in single-image mode
        callback=single_image_callback,
        **kwargs,
    )

    # # Calculate overall recall after processing all images
    calculate_overall_recall(evaluation_results)

    print("Finished evaluation with custom callback.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to evaluate.")
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--output_dir", type=Path, help="Directory to save visualization and results.")
    parser.add_argument("--num", type=int, help="Number of images to process.")
    parser.add_argument("dotlist", nargs="*", help="Override config params from command line.")
    args = parser.parse_args()

    cfg = OmegaConf.from_cli(args.dotlist)

    # Prepare kwargs for the run function
    kwargs = vars(args)
    kwargs['cfg'] = cfg
    kwargs.pop('dotlist', None)  # Already processed into cfg
    kwargs.pop('sequential', None)  # No longer used
    run(**kwargs)
