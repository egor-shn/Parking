import cv2
import json
import sys
import os
import numpy as np
from ultralytics import YOLO

# Добавляем корневую папку в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    create_spot_mask,
    mask_overlap_ratio,
    resize_mask_to_frame,
    rect_intersects_polygon
)


class ParkingDetector:
    def __init__(self, config_path='config/parking_spots.json', model_name='yolo11m-seg.pt', overlap_threshold=0.3):
        """
        Инициализация детектора с Pixel-wise ROI и диагностикой.
        
        Args:
            config_path: путь к файлу с зонами парковочных мест
            model_name: название модели YOLO
            overlap_threshold: порог перекрытия маски для определения занятости (0.0 - 1.0)
        """
        # Загружаем конфигурацию с зонами мест
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.spots = self.config['spots']
        self.total_spots = len(self.spots)
        self.overlap_threshold = overlap_threshold
        
        # Загружаем модель YOLO11m
        print("🚀 Загрузка модели YOLO11m...")
        self.model = YOLO(model_name)
        print("✅ Модель загружена")
        
        # Кэш для масок мест (оптимизация)
        self.spot_mask_cache = {}
    
    def detect_vehicles_with_masks(self, image):
        """
        Детектирует транспорт на изображении с получением масок.
        """
        results = self.model(image, imgsz=1280, retina_masks=True)
        vehicles = []
        
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if boxes is not None and masks is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    mask = masks[i].data[0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    mask_resized = resize_mask_to_frame(mask, image.shape)
                    
                    if class_id in [2, 3, 5, 7] and confidence > 0.3:
                        vehicles.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'mask': mask_resized,
                            'mask_area': np.sum(mask_resized)
                        })
            elif boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in [2, 3, 5, 7] and confidence > 0.3:
                        vehicles.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'mask': None,
                            'mask_area': 0
                        })
        
        return vehicles
    
    def get_spot_mask(self, spot, frame_shape):
        """Возвращает маску для парковочного места (с кэшированием)."""
        spot_id = spot['id']
        if spot_id not in self.spot_mask_cache:
            self.spot_mask_cache[spot_id] = create_spot_mask(spot['polygon'], frame_shape)
        return self.spot_mask_cache[spot_id]
    
    def analyze_spot_with_diagnostics(self, spot, vehicles, frame_shape):
        """
        Анализирует одно парковочное место с диагностикой.
        
        Returns:
            dict: {
                'occupied': bool,
                'reason': str,
                'overlap_ratio': float,
                'best_match': dict or None
            }
        """
        spot_mask = self.get_spot_mask(spot, frame_shape)
        spot_area = np.sum(spot_mask)
        
        # Если место вне кадра
        if spot_area == 0:
            return {
                'occupied': False,
                'reason': 'spot_out_of_frame',
                'overlap_ratio': 0.0,
                'best_match': None
            }
        
        best_match = None
        best_overlap = 0.0
        best_vehicle = None
        
        for vehicle in vehicles:
            if vehicle['mask'] is not None:
                overlap = mask_overlap_ratio(vehicle['mask'], spot_mask)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vehicle = vehicle
            else:
                # Fallback на прямоугольники
                if rect_intersects_polygon(vehicle, spot['polygon']):
                    if best_vehicle is None or 1.0 > best_overlap:
                        best_overlap = 1.0
                        best_vehicle = vehicle
        
        # Определяем причину
        is_occupied = best_overlap > self.overlap_threshold
        
        if is_occupied:
            reason = f"перекрытие {best_overlap*100:.1f}% > порог {self.overlap_threshold*100:.0f}%"
            if best_vehicle:
                reason += f" (класс: {best_vehicle.get('class_name', 'unknown')}, уверенность: {best_vehicle.get('confidence', 0):.2f})"
        else:
            if best_vehicle:
                reason = f"перекрытие {best_overlap*100:.1f}% < порог {self.overlap_threshold*100:.0f}%"
            else:
                reason = "нет транспортных средств в зоне"
        
        return {
            'occupied': is_occupied,
            'reason': reason,
            'overlap_ratio': best_overlap,
            'best_match': best_vehicle,
            'spot_area': spot_area
        }
    
    def analyze_image(self, image_path):
        """
        Анализирует изображение и определяет статус каждого места с диагностикой.
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Очищаем кэш масок для нового изображения
        self.spot_mask_cache = {}
        
        # Детектируем транспорт с масками
        vehicles = self.detect_vehicles_with_masks(image)
        print(f"🚗 Найдено транспорта: {len(vehicles)}")
        
        # Выводим информацию о найденном транспорте
        for v in vehicles:
            print(f"   {v['class_name']}: уверенность {v['confidence']:.2f}, маска {v['mask_area']} пикс.")
        
        # Анализируем каждое место с диагностикой
        results = []
        for spot in self.spots:
            spot_id = spot['id']
            diag = self.analyze_spot_with_diagnostics(spot, vehicles, image.shape)
            
            results.append({
                'id': spot_id,
                'occupied': diag['occupied'],
                'diagnostic': diag
            })
        
        # Подсчитываем количество свободных мест
        total_free = sum(1 for r in results if not r['occupied'])
        
        return {
            'total_spots': self.total_spots,
            'free_spots': total_free,
            'occupied_spots': self.total_spots - total_free,
            'vehicles_detected': len(vehicles),
            'spots': results
        }
    
    def visualize(self, image_path, results):
        """
        Визуализирует результаты анализа на изображении с диагностикой.
        """
        image = cv2.imread(image_path)
        
        # Рисуем зоны мест
        for spot_result in results['spots']:
            spot_id = spot_result['id']
            is_occupied = spot_result['occupied']
            diag = spot_result['diagnostic']
            
            spot_data = next(s for s in self.spots if s['id'] == spot_id)
            polygon = np.array(spot_data['polygon'], dtype=np.int32)
            
            # Цвет: красный - занято, зеленый - свободно
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            
            # Рисуем полигон
            cv2.polylines(image, [polygon], True, color, 2)
            
            # Добавляем номер места
            centroid = np.mean(polygon, axis=0).astype(int)
            cv2.putText(image, str(spot_id), 
                       (centroid[0] - 10, centroid[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Добавляем диагностику (если место занято)
            if is_occupied and diag['best_match']:
                # Рисуем крестик в центре места
                cv2.drawMarker(image, (centroid[0], centroid[1]), (0, 0, 255), 
                               cv2.MARKER_CROSS, 20, 2)
        
        # Добавляем общую информацию
        info = f"Свободно: {results['free_spots']} / {results['total_spots']}"
        cv2.putText(image, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(image, f"Транспорт: {results['vehicles_detected']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Сохраняем результат
        output_path = 'data/result_yolo11m_pixelwise.jpg'
        cv2.imwrite(output_path, image)
        print(f"📸 Результат сохранен в: {output_path}")
        
        return image
    
    def print_diagnostics(self, results):
        """
        Выводит подробную диагностику по каждому месту.
        """
        print("\n" + "=" * 80)
        print("📊 ДИАГНОСТИКА ПО МЕСТАМ")
        print("=" * 80)
        print(f"{'ID':>4} | {'Статус':>10} | {'Причина':<60}")
        print("-" * 80)
        
        error_count = 0
        for spot in results['spots']:
            spot_id = spot['id']
            status = "ЗАНЯТО" if spot['occupied'] else "СВОБОДНО"
            diag = spot['diagnostic']
            reason = diag['reason']
            
            # Отмечаем подозрительные места
            if spot['occupied'] and diag['overlap_ratio'] < 0.5:
                reason = "⚠️ " + reason + " (низкое перекрытие!)"
                error_count += 1
            elif not spot['occupied'] and diag['best_match'] is not None:
                reason = "⚠️ " + reason + " (есть транспорт, но порог не достигнут!)"
                error_count += 1
            
            print(f"{spot_id:>4} | {status:>10} | {reason:<60}")
        
        print("-" * 80)
        print(f"Всего мест: {results['total_spots']}")
        print(f"Свободных: {results['free_spots']}")
        print(f"Занятых: {results['occupied_spots']}")
        if error_count > 0:
            print(f"⚠️ Подозрительных мест: {error_count}")
        print("=" * 80)


def main():
    # Создаем детектор
    detector = ParkingDetector(
        config_path='config/parking_spots.json',
        model_name='yolo11m-seg.pt',
        overlap_threshold=0.3
    )
    
    # Анализируем изображение
    image_path = 'data/test_image.jpg'
    results = detector.analyze_image(image_path)
    
    # Выводим диагностику
    detector.print_diagnostics(results)
    
    # Визуализируем результат
    detector.visualize(image_path, results)


if __name__ == "__main__":
    main()