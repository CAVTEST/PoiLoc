#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始OSM数据处理模块
直接处理OSM原始数据，不使用语义分类，用于POI匹配
"""
import sys

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
from dataclasses import dataclass


sys.path.append(str(Path(__file__).resolve().parent.parent))
from maploc.utils.geo import BoundaryBox, Projection
from maploc.osm.download import get_osm
from maploc.osm.reader import OSMData


@dataclass
class RawOSMPOI:
    node_id: str
    name: str
    category: str
    subcategory: str
    tags: Dict[str, str]
    lat: float
    lon: float
    xy: Optional[Tuple[float, float]] = None


class RawOSMProcessor:


    def __init__(self):

        self.poi_categories = {

            'food': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'food_court'],
            'restaurant': ['restaurant', 'fast_food', 'food_court'],
            'cafe': ['cafe', 'coffee_shop'],
            'bar': ['bar', 'pub', 'nightclub'],


            'accommodation': ['hotel', 'motel', 'hostel', 'guest_house', 'apartment'],
            'hotel': ['hotel', 'motel', 'hostel', 'guest_house', 'apartment'],


            'shop': ['shop', 'supermarket', 'convenience', 'mall', 'department_store'],
            'supermarket': ['supermarket', 'convenience'],


            'finance': ['bank', 'atm'],
            'healthcare': ['hospital', 'clinic', 'pharmacy', 'dentist'],
            'education': ['school', 'university', 'college', 'kindergarten'],
            'government': ['police', 'fire_station', 'post_office'],


            'transport': ['fuel', 'charging_station', 'parking', 'bus_stop', 'traffic_signals',
                          'stop', 'give_way', 'crossing', 'speed_camera', 'street_lamp'],
            'fuel': ['fuel', 'charging_station'],
            'parking': ['parking'],

            'entertainment': ['cinema', 'theatre'],
            'sports': ['gym', 'fitness_centre'],

            'infrastructure': ['street_lamp', 'lighting'],


            'religion': ['place_of_worship'],
            'library': ['library'],
            'tourism': ['tourism', 'attraction', 'museum', 'gallery']
        }

        self.tag_to_category = {}
        for category, tags in self.poi_categories.items():
            for tag in tags:
                self.tag_to_category[tag] = category

        special_mappings = {
            'traffic_signals': 'transport',
            'stop': 'transport',
            'give_way': 'transport',
            'crossing': 'transport',
            'bus_stop': 'transport',
            'speed_camera': 'transport',
            'street_lamp': 'infrastructure',
            'lighting': 'infrastructure',
            'railway_station': 'transport',
            'level_crossing': 'transport',
            'public_transport': 'transport',
            'traffic_sign': 'transport',
            'place_of_worship': 'religion'
        }
        self.tag_to_category.update(special_mappings)

    def extract_pois_from_osm(self, osm_data: OSMData, projection: Projection,
                              bbox_xy: BoundaryBox) -> List[RawOSMPOI]:

        pois = []

        for node_id, node in osm_data.nodes.items():
            if not node.tags:
                continue

            poi_info = self._extract_poi_info(node)
            if poi_info is None:
                continue

            name, category, subcategory = poi_info

            try:
                latlon = node.geo  # [lat, lon]
                xy = projection.project(latlon.reshape(1, -1))[0]


                if bbox_xy.contains(xy):
                    poi = RawOSMPOI(
                        node_id=str(node_id),
                        name=name,
                        category=category,
                        subcategory=subcategory,
                        tags=dict(node.tags),
                        lat=float(latlon[0]),
                        lon=float(latlon[1]),
                        xy=(float(xy[0]), float(xy[1]))
                    )
                    pois.append(poi)

            except Exception as e:

                continue

        crossing_count = 0
        for way_id, way in osm_data.ways.items():
            if way.tags:
                if (way.tags.get('highway') == 'footway' and
                    way.tags.get('footway') == 'crossing') or \
                        (way.tags.get('highway') == 'crossing'):
                    try:

                        if way.nodes and len(way.nodes) > 0:
                            mid_idx = len(way.nodes) // 2
                            mid_node = way.nodes[mid_idx]
                            if mid_node and hasattr(mid_node, 'geo'):
                                latlon = mid_node.geo
                                xy = projection.project(latlon.reshape(1, -1))[0]

                                if bbox_xy.contains(xy):

                                    poi = RawOSMPOI(
                                        node_id=f"way_{way_id}_crossing",
                                        name="crossing",
                                        category="transport",
                                        subcategory="crossing",
                                        tags=dict(way.tags),
                                        lat=float(latlon[0]),
                                        lon=float(latlon[1]),
                                        xy=(float(xy[0]), float(xy[1]))
                                    )
                                    pois.append(poi)
                                    crossing_count += 1
                    except:
                        continue

        print(f"Extracted {len(pois)} POIs from OSM data (including {crossing_count} crossings)")
        return pois

    def _extract_poi_info(self, node) -> Optional[Tuple[str, str, str]]:

        tags = node.tags
        if not tags:
            return None


        name = tags.get('name', '')

        if 'highway' in tags:
            highway = tags['highway']
            if highway in ['traffic_signals', 'stop', 'give_way', 'crossing',
                           'bus_stop', 'speed_camera', 'street_lamp']:
                display_name = name or highway.replace('_', ' ')
                return display_name, 'transport', highway

        elif 'traffic_sign' in tags:
            display_name = name or 'traffic_sign'
            return display_name, 'transport', 'traffic_sign'

        elif 'railway' in tags:
            railway = tags['railway']
            if railway == 'station':
                display_name = name or 'railway_station'
                return display_name, 'transport', railway
            elif railway == 'level_crossing':
                display_name = name or 'level_crossing'
                return display_name, 'transport', railway

        elif 'public_transport' in tags:
            display_name = name or 'public_transport'
            return display_name, 'transport', 'public_transport'

        elif 'man_made' in tags and tags['man_made'] == 'street_lamp':
            display_name = name or 'street_lamp'
            return display_name, 'infrastructure', 'street_lamp'
        elif 'amenity' in tags and tags['amenity'] == 'lighting':
            display_name = name or 'lighting'
            return display_name, 'infrastructure', 'lighting'

        if not name:
            return None


        if 'amenity' in tags:
            amenity = tags['amenity']
            if amenity in ['restaurant', 'cafe', 'bar', 'pub', 'fast_food']:
                return name, 'food', amenity
            elif amenity in ['bank', 'atm']:
                return name, 'finance', amenity
            elif amenity in ['hospital', 'clinic', 'pharmacy']:
                return name, 'healthcare', amenity
            elif amenity in ['school', 'university', 'college', 'library']:
                return name, 'education', amenity
            elif amenity in ['fuel', 'charging_station']:
                return name, 'fuel', amenity
            elif amenity in ['police', 'fire_station', 'post_office']:
                return name, 'government', amenity
            elif amenity in ['parking']:
                return name, 'transport', amenity
            elif amenity in ['cinema', 'theatre']:
                return name, 'entertainment', amenity
            elif amenity in ['gym', 'fitness_centre']:
                return name, 'sports', amenity
            elif amenity in ['place_of_worship']:
                return name, 'religion', amenity

        elif 'shop' in tags:
            shop = tags['shop']
            return name, 'shop', shop

        elif 'tourism' in tags:
            tourism = tags['tourism']
            if tourism in ['hotel', 'motel', 'hostel', 'guest_house', 'apartment']:
                return name, 'accommodation', tourism
            return name, 'tourism', tourism

        if name:
            return name, 'other', 'unknown'

        return None

    def _get_default_name(self, tags: Dict[str, str]) -> str:

        if 'highway' in tags:
            highway = tags['highway']
            if highway in ['traffic_signals', 'stop', 'give_way', 'crossing',
                           'bus_stop', 'speed_camera', 'street_lamp']:
                return highway.replace('_', ' ')

        if 'traffic_sign' in tags:
            return 'traffic_sign'


        if 'railway' in tags:
            railway = tags['railway']
            if railway in ['station', 'level_crossing']:
                return railway.replace('_', ' ')

        if 'public_transport' in tags:
            return 'public_transport'

        if 'man_made' in tags and tags['man_made'] == 'street_lamp':
            return 'street_lamp'
        if 'amenity' in tags and tags['amenity'] == 'lighting':
            return 'lighting'

        if 'amenity' in tags:
            amenity = tags['amenity']
            if amenity in ['fuel', 'charging_station', 'parking', 'atm']:
                return amenity.replace('_', ' ')

        return ''

    def _determine_category(self, tags: Dict[str, str]) -> Tuple[str, str]:

        if 'highway' in tags:
            highway = tags['highway']
            if highway in ['traffic_signals', 'stop', 'give_way', 'crossing',
                           'bus_stop', 'speed_camera', 'street_lamp']:
                return 'transport', highway


        if 'traffic_sign' in tags:
            return 'transport', 'traffic_sign'


        if 'railway' in tags:
            railway = tags['railway']
            if railway in ['station', 'level_crossing']:
                return 'transport', railway

        if 'public_transport' in tags:
            return 'transport', 'public_transport'


        if 'man_made' in tags and tags['man_made'] == 'street_lamp':
            return 'infrastructure', 'street_lamp'

        if 'amenity' in tags:
            amenity = tags['amenity']
            if amenity in ['restaurant', 'cafe', 'bar', 'pub', 'fast_food']:
                return 'food', amenity
            elif amenity in ['bank', 'atm']:
                return 'finance', amenity
            elif amenity in ['hospital', 'clinic', 'pharmacy']:
                return 'healthcare', amenity
            elif amenity in ['school', 'university', 'college', 'library']:
                return 'education', amenity
            elif amenity in ['fuel', 'charging_station']:
                return 'fuel', amenity
            elif amenity in ['police', 'fire_station', 'post_office']:
                return 'government', amenity
            elif amenity in ['parking']:
                return 'transport', amenity
            elif amenity in ['cinema', 'theatre']:
                return 'entertainment', amenity
            elif amenity in ['gym', 'fitness_centre']:
                return 'sports', amenity
            elif amenity in ['place_of_worship']:
                return 'religion', amenity
            elif amenity in ['lighting']:
                return 'infrastructure', amenity

        if 'shop' in tags:
            shop = tags['shop']
            return 'shop', shop

        if 'tourism' in tags:
            tourism = tags['tourism']
            if tourism in ['hotel', 'motel', 'hostel', 'guest_house', 'apartment']:
                return 'accommodation', tourism
            return 'tourism', tourism


        for tag_key, tag_value in tags.items():
            category = self.tag_to_category.get(tag_value)
            if category:
                return category, tag_value

        return '', ''

    def match_yolo_pois_with_osm(self, yolo_detections: List[Dict],
                                 osm_pois: List[RawOSMPOI],
                                 max_distance: float = 100.0) -> List[Dict]:

        matches = []

        for detection in yolo_detections:
            yolo_class = detection['class_name'].lower()


            candidate_pois = []
            for poi in osm_pois:
                if self._is_category_match(yolo_class, poi.category, poi.subcategory):
                    candidate_pois.append(poi)

            if not candidate_pois:
                print(f"No OSM POI candidates found for YOLO detection: {yolo_class}")
                continue


            if len(candidate_pois) == 1:
                match = {
                    'yolo_detection': detection,
                    'osm_poi': candidate_pois[0],
                    'match_type': 'unique',
                    'confidence': 1.0,
                    'distance': 0.0
                }
                matches.append(match)
                print(f"Unique match found: {yolo_class} -> {candidate_pois[0].name}")
            else:

                match = {
                    'yolo_detection': detection,
                    'osm_candidates': candidate_pois,
                    'match_type': 'multiple',
                    'confidence': 0.5,
                    'distance': 0.0
                }
                matches.append(match)
                print(f"Multiple candidates found for {yolo_class}: {[poi.name for poi in candidate_pois]}")

        return matches

    def normalize_string(self, s: str) -> str:
        return s.strip().replace('’', "'").lower()

    def _is_category_match(self, yolo_class: str, osm_category: str, osm_subcategory: str) -> bool:

        yolo_class = self.normalize_string(yolo_class)
        osm_category = self.normalize_string(osm_category)
        osm_subcategory = self.normalize_string(osm_subcategory)


        if yolo_class == osm_category or yolo_class == osm_subcategory:
            return True


        yolo_words = set(yolo_class.split('_'))
        osm_words = set(osm_category.split('_') + osm_subcategory.split('_'))

        if yolo_words & osm_words:
            return True

        special_matches = {
            'coffee': ['cafe', 'coffee_shop', 'food'],
            'restaurant': ['restaurant', 'fast_food', 'food_court', 'food'],
            'hotel': ['hotel', 'motel', 'hostel', 'guest_house', 'accommodation'],
            'shop': ['shop', 'supermarket', 'convenience', 'mall'],
            'bank': ['bank', 'atm', 'finance'],
            'hospital': ['hospital', 'clinic', 'pharmacy', 'healthcare'],
            'school': ['school', 'university', 'college', 'education'],
            'fuel': ['fuel', 'charging_station'],
            'parking': ['parking', 'transport'],
            'traffic': ['traffic_signals', 'stop', 'give_way', 'crossing', 'transport'],
            'lamp': ['street_lamp', 'lighting', 'infrastructure'],
            'cinema': ['cinema', 'theatre', 'entertainment'],
            'gym': ['gym', 'fitness_centre', 'sports'],
            'church': ['place_of_worship', 'religion'],
            'library': ['library', 'education'],
            'police': ['police', 'government'],
            'fire': ['fire_station', 'government'],
            'lamp': ['street lamp']
        }

        if yolo_class in special_matches:
            return any(match in osm_subcategory or match == osm_category
                       for match in special_matches[yolo_class])

        return False

    def refine_matches_with_orientation(self, matches: List[Dict],
                                        camera_yaw: float,
                                        image_center_xy: Tuple[float, float]) -> List[Dict]:

        refined_matches = []

        for match in matches:
            if match['match_type'] == 'unique':

                refined_matches.append(match)
            elif match['match_type'] == 'multiple':

                best_poi = self._select_best_poi_by_orientation(
                    match['osm_candidates'],
                    camera_yaw,
                    image_center_xy,
                    match['yolo_detection']
                )

                if best_poi:
                    refined_match = {
                        'yolo_detection': match['yolo_detection'],
                        'osm_poi': best_poi,
                        'match_type': 'orientation_refined',
                        'confidence': 0.8,
                        'distance': self._calculate_distance(image_center_xy, best_poi.xy)
                    }
                    refined_matches.append(refined_match)
                    print(f"Orientation-refined match: {match['yolo_detection']['class_name']} -> {best_poi.name}")

        return refined_matches

    def _select_best_poi_by_orientation(self, candidates: List[RawOSMPOI],
                                        camera_yaw: float,
                                        camera_xy: Tuple[float, float],
                                        yolo_detection: Dict) -> Optional[RawOSMPOI]:

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]


        best_poi = None
        best_score = -1

        for poi in candidates:
            if poi.xy is None:
                continue


            dx = poi.xy[0] - camera_xy[0]
            dy = poi.xy[1] - camera_xy[1]
            poi_angle = np.degrees(np.arctan2(dx, dy))
            angle_diff = abs(self._normalize_angle(poi_angle - camera_yaw))


            distance = np.sqrt(dx ** 2 + dy ** 2)
            distance_score = 1.0 / (1.0 + distance / 50.0)


            angle_score = 1.0 - angle_diff / 180.0


            total_score = 0.6 * angle_score + 0.4 * distance_score

            if total_score > best_score:
                best_score = total_score
                best_poi = poi

        return best_poi

    def _normalize_angle(self, angle: float) -> float:

        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _calculate_distance(self, xy1: Tuple[float, float], xy2: Tuple[float, float]) -> float:

        if xy2 is None:
            return float('inf')
        return np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

    def get_location_from_matches(self, matches: List[Dict]) -> Optional[Tuple[float, float]]:

        if not matches:
            return None


        if len(matches) == 1:
            poi = matches[0]['osm_poi']
            return (poi.lat, poi.lon)


        total_weight = 0
        weighted_lat = 0
        weighted_lon = 0

        for match in matches:
            poi = match['osm_poi']
            weight = match['confidence']

            weighted_lat += poi.lat * weight
            weighted_lon += poi.lon * weight
            total_weight += weight

        if total_weight > 0:
            avg_lat = weighted_lat / total_weight
            avg_lon = weighted_lon / total_weight
            return (avg_lat, avg_lon)

        return None

    def save_matches_to_json(self, matches: List[Dict], output_path: str):

        serializable_matches = []

        for match in matches:
            serializable_match = {
                'yolo_detection': match['yolo_detection'],
                'match_type': match['match_type'],
                'confidence': match['confidence'],
                'distance': match['distance']
            }

            if 'osm_poi' in match:
                poi = match['osm_poi']
                serializable_match['osm_poi'] = {
                    'node_id': poi.node_id,
                    'name': poi.name,
                    'category': poi.category,
                    'subcategory': poi.subcategory,
                    'tags': poi.tags,
                    'lat': poi.lat,
                    'lon': poi.lon,
                    'xy': poi.xy
                }

            if 'osm_candidates' in match:
                serializable_match['osm_candidates'] = []
                for poi in match['osm_candidates']:
                    serializable_match['osm_candidates'].append({
                        'node_id': poi.node_id,
                        'name': poi.name,
                        'category': poi.category,
                        'subcategory': poi.subcategory,
                        'tags': poi.tags,
                        'lat': poi.lat,
                        'lon': poi.lon,
                        'xy': poi.xy
                    })

            serializable_matches.append(serializable_match)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_matches, f, indent=2, ensure_ascii=False)

        print(f"Matches saved to {output_path}")


def load_osm_data_for_location(lat: float, lon: float,
                               map_size_meters: int = 256) -> Tuple[OSMData, Projection, BoundaryBox]:


    projection = Projection(lat, lon)


    half_size = map_size_meters / 2
    bbox_xy = BoundaryBox(
        np.array([-half_size, -half_size]),
        np.array([half_size, half_size])
    )


    bbox_osm = projection.unproject(bbox_xy + 10)
    osm_file_path = get_osm(bbox_osm, overwrite=False, proxy_url="http://127.0.0.1:10808")

    if not osm_file_path:
        raise RuntimeError("Failed to download OSM data")


    osm_data = OSMData.from_file(Path(osm_file_path))
    osm_data.add_xy_to_nodes(projection)

    return osm_data, projection, bbox_xy


if __name__ == "__main__":

    processor = RawOSMProcessor()


    test_lat, test_lon = 48.8566, 2.3522

    try:
        osm_data, projection, bbox_xy = load_osm_data_for_location(test_lat, test_lon)
        pois = processor.extract_pois_from_osm(osm_data, projection, bbox_xy)

        print(f"Found {len(pois)} POIs:")
        for poi in pois[:10]:
            print(f"  - {poi.name} ({poi.category}/{poi.subcategory})")

    except Exception as e:
        print(f"Test failed: {e}")