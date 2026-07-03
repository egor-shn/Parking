import cv2
import numpy as np
from shapely.geometry import Polygon, box


def point_in_polygon(point, polygon):
    """Проверяет, находится ли точка внутри полигона."""
    poly = Polygon(polygon)
    return poly.contains(point)


def boxes_intersect(box1, box2):
    """Проверяет пересечение двух прямоугольников."""
    b1 = box(box1['x1'], box1['y1'], box1['x2'], box1['y2'])
    b2 = box(box2['x1'], box2['y1'], box2['x2'], box2['y2'])
    return b1.intersects(b2)


def rect_intersects_polygon(rect, polygon):
    """Проверяет, пересекается ли прямоугольник автомобиля с полигоном места."""
    rect_box = box(rect['x1'], rect['y1'], rect['x2'], rect['y2'])
    poly = Polygon(polygon)
    return rect_box.intersects(poly)


def get_intersection_area(rect, polygon):
    """Возвращает площадь пересечения прямоугольника и полигона."""
    rect_box = box(rect['x1'], rect['y1'], rect['x2'], rect['y2'])
    poly = Polygon(polygon)
    intersection = rect_box.intersection(poly)
    return intersection.area


def create_spot_mask(polygon, frame_shape):
    """
    Создаёт бинарную маску для зоны парковочного места.
    
    Args:
        polygon: список точек [(x1,y1), (x2,y2), ...]
        frame_shape: (height, width) исходного кадра
        
    Returns:
        np.ndarray: бинарная маска (0/1)
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_overlap_ratio(vehicle_mask, spot_mask):
    """
    Вычисляет коэффициент перекрытия маски машины и маски места.
    
    Args:
        vehicle_mask: бинарная маска машины (0/1)
        spot_mask: бинарная маска места (0/1)
        
    Returns:
        float: доля перекрытия (0.0 - 1.0)
    """
    # Приводим маски к одному размеру
    if vehicle_mask.shape != spot_mask.shape:
        # Масштабируем маску машины до размера маски места
        vehicle_mask = cv2.resize(
            vehicle_mask, 
            (spot_mask.shape[1], spot_mask.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    # Вычисляем пересечение
    intersection = np.logical_and(vehicle_mask, spot_mask)
    overlap_pixels = np.sum(intersection)
    
    # Площадь места
    spot_area = np.sum(spot_mask)
    
    if spot_area == 0:
        return 0.0
    
    return overlap_pixels / spot_area


def resize_mask_to_frame(mask, target_shape):
    """
    Масштабирует маску до размера кадра.
    
    Args:
        mask: маска от YOLO
        target_shape: (height, width) целевого кадра
        
    Returns:
        np.ndarray: масштабированная маска
    """
    if mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(
            mask.astype(np.float32),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return (mask > 0.5).astype(np.uint8)