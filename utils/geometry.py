import cv2
import numpy as np
from shapely.geometry import Polygon, box


def point_in_polygon(point, polygon):
    poly = Polygon(polygon)
    return poly.contains(point)


def boxes_intersect(box1, box2):
    b1 = box(box1['x1'], box1['y1'], box1['x2'], box1['y2'])
    b2 = box(box2['x1'], box2['y1'], box2['x2'], box2['y2'])
    return b1.intersects(b2)


def rect_intersects_polygon(rect, polygon):
    rect_box = box(rect['x1'], rect['y1'], rect['x2'], rect['y2'])
    poly = Polygon(polygon)
    return rect_box.intersects(poly)


def get_intersection_area(rect, polygon):
    rect_box = box(rect['x1'], rect['y1'], rect['x2'], rect['y2'])
    poly = Polygon(polygon)
    intersection = rect_box.intersection(poly)
    return intersection.area


def create_spot_mask(polygon, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_overlap_ratio(vehicle_mask, spot_mask):
    if vehicle_mask.shape != spot_mask.shape:
        vehicle_mask = cv2.resize(
            vehicle_mask, 
            (spot_mask.shape[1], spot_mask.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    intersection = np.logical_and(vehicle_mask, spot_mask)
    overlap_pixels = np.sum(intersection)
    
    spot_area = np.sum(spot_mask)
    
    if spot_area == 0:
        return 0.0
    
    return overlap_pixels / spot_area


def resize_mask_to_frame(mask, target_shape):
    if mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(
            mask.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return (mask > 0.5).astype(np.uint8)
