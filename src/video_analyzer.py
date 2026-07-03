import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import (
    create_spot_mask,
    mask_overlap_ratio,
    resize_mask_to_frame
)


class VideoParkingAnalyzer:
    def __init__(self, video_path, config_path='config/parking_spots.json',
                 model_name='yolo11m.pt', interval=2, save_video=False, overlap_threshold=0.3):
        """
        Анализатор парковки на видеофайле с Pixel-wise ROI.
        
        Args:
            video_path: путь к видеофайлу
            config_path: путь к конфигурации с зонами мест
            model_name: название модели YOLO
            interval: интервал между анализами в секундах
            save_video: сохранять ли видео с результатами
            overlap_threshold: порог перекрытия маски (0.0 - 1.0)
        """
        self.video_path = video_path
        self.interval = interval
        self.save_video = save_video
        self.overlap_threshold = overlap_threshold
        
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.spots = self.config['spots']
        self.total_spots = len(self.spots)
        
        # Загружаем модель YOLO11m
        print("🚀 Загрузка модели YOLO11m...")
        self.model = YOLO(model_name)
        print(f"✅ Модель загружена: {model_name}")
        
        # Открываем видео
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.connect_video()
        
        # Для сохранения видео
        self.video_writer = None
        
        # Кэш масок мест
        self.spot_mask_cache = {}
        
        # Статистика
        self.frame_count = 0
        self.processed_count = 0
        self.last_result = None
    
    def connect_video(self):
        """Открывает видеофайл"""
        print(f"🎬 Открытие видео: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Видео открыто")
        print(f"   Разрешение: {self.video_width}x{self.video_height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Кадров: {self.total_frames}")
        print(f"   Длительность: {self.total_frames / self.fps / 60:.1f} мин")
    
    def get_spot_mask(self, spot, frame_shape):
        """Возвращает маску для парковочного места (с кэшированием)."""
        spot_id = spot['id']
        if spot_id not in self.spot_mask_cache:
            self.spot_mask_cache[spot_id] = create_spot_mask(spot['polygon'], frame_shape)
        return self.spot_mask_cache[spot_id]
    
    def detect_vehicles_with_masks(self, frame):
        """
        Детектирует транспорт на кадре с получением масок.
        """
        results = self.model(frame, imgsz=1280, retina_masks=True)
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
                    mask_resized = resize_mask_to_frame(mask, frame.shape)
                    
                    if class_id in [2, 3, 5, 7] and confidence > 0.3:
                        vehicles.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'mask': mask_resized
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
                            'mask': None
                        })
        
        return vehicles
    
    def analyze_spot_with_mask(self, spot, vehicles, frame_shape):
        """Анализирует одно парковочное место с использованием масок."""
        spot_mask = self.get_spot_mask(spot, frame_shape)
        spot_area = np.sum(spot_mask)
        
        if spot_area == 0:
            return False
        
        for vehicle in vehicles:
            if vehicle['mask'] is not None:
                overlap = mask_overlap_ratio(vehicle['mask'], spot_mask)
                if overlap > self.overlap_threshold:
                    return True
            else:
                from utils.geometry import rect_intersects_polygon
                if rect_intersects_polygon(vehicle, spot['polygon']):
                    return True
        
        return False
    
    def analyze_frame(self, frame):
        """Анализирует один кадр."""
        # Очищаем кэш масок для нового кадра
        self.spot_mask_cache = {}
        
        # Детектируем транспорт с масками
        vehicles = self.detect_vehicles_with_masks(frame)
        
        # Анализируем каждое место
        results = []
        for spot in self.spots:
            spot_id = spot['id']
            is_occupied = self.analyze_spot_with_mask(spot, vehicles, frame.shape)
            
            results.append({
                'id': spot_id,
                'occupied': is_occupied,
                'vehicle': None
            })
        
        total_free = sum(1 for r in results if not r['occupied'])
        
        return {
            'total_spots': self.total_spots,
            'free_spots': total_free,
            'occupied_spots': self.total_spots - total_free,
            'vehicles_detected': len(vehicles),
            'spots': results,
            'timestamp': time.time()
        }
    
    def visualize(self, frame, results):
        """Визуализирует результаты на кадре."""
        vis_frame = frame.copy()
        
        # Рисуем зоны мест
        for spot_result in results['spots']:
            spot_id = spot_result['id']
            is_occupied = spot_result['occupied']
            
            spot_data = next(s for s in self.spots if s['id'] == spot_id)
            polygon = np.array(spot_data['polygon'], dtype=np.int32)
            
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.polylines(vis_frame, [polygon], True, color, 2)
            
            centroid = np.mean(polygon, axis=0).astype(int)
            cv2.putText(vis_frame, str(spot_id),
                       (centroid[0] - 10, centroid[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Информация на экране
        info = f"Свободно: {results['free_spots']} / {results['total_spots']}"
        cv2.putText(vis_frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(vis_frame, f"Транспорт: {results['vehicles_detected']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        time_str = time.strftime("%H:%M:%S")
        cv2.putText(vis_frame, time_str, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(vis_frame, f"Кадр: {self.frame_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return vis_frame
    
    def run(self):
        """Основной цикл обработки видео."""
        print("\n" + "=" * 60)
        print("🚗 АНАЛИЗ ПАРКОВКИ (YOLO11m + Pixel-wise ROI)")
        print("=" * 60)
        print(f"Видео: {self.video_path}")
        print(f"Интервал: {self.interval} сек")
        print(f"Всего мест: {self.total_spots}")
        print(f"Порог перекрытия: {self.overlap_threshold:.0%}")
        print("=" * 60)
        print("Нажмите 'q' для выхода")
        print("Нажмите 'p' для паузы/продолжения")
        print("=" * 60)
        
        # Настройка сохранения видео
        if self.save_video:
            output_path = 'data/video_result_pixelwise.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps,
                (self.video_width, self.video_height)
            )
            print(f"📹 Сохранение видео в: {output_path}")
        
        paused = False
        
        try:
            while True:
                cycle_start = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\n🔄 Видео закончилось, перемотка...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                
                self.frame_count += 1
                
                if not paused:
                    if self.frame_count % max(1, int(self.fps * self.interval)) == 0:
                        results = self.analyze_frame(frame)
                        self.processed_count += 1
                        self.last_result = results
                        
                        print(f"\n📊 [{time.strftime('%H:%M:%S')}] "
                              f"Свободно: {results['free_spots']}/{results['total_spots']} "
                              f"(транспорта: {results['vehicles_detected']})")
                    else:
                        results = self.last_result
                        if results is None:
                            results = self.analyze_frame(frame)
                            self.processed_count += 1
                            self.last_result = results
                else:
                    results = self.last_result
                    if results is None:
                        results = self.analyze_frame(frame)
                        self.processed_count += 1
                        self.last_result = results
                
                if results is not None:
                    vis_frame = self.visualize(frame, results)
                else:
                    vis_frame = frame
                
                cv2.imshow('Parking Analyzer (Pixel-wise ROI)', vis_frame)
                
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Завершение работы...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Пауза' if paused else '▶️  Продолжение'}")
                elif key == ord('s'):
                    cv2.imwrite('data/screenshot_pixelwise.jpg', vis_frame)
                    print("📸 Скриншот сохранён в data/screenshot_pixelwise.jpg")
                
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.interval - elapsed)
                if not paused and sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n👋 Прервано пользователем")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Освобождение ресурсов"""
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА (YOLO11m + Pixel-wise ROI)")
        print("=" * 60)
        print(f"Всего кадров обработано: {self.frame_count}")
        print(f"Анализов выполнено: {self.processed_count}")
        if self.last_result is not None:
            print(f"Финальный статус: свободно {self.last_result['free_spots']}/{self.last_result['total_spots']}")
        print("=" * 60)
        print("✅ Ресурсы освобождены")


def main():
    # Настройки
    VIDEO_PATH = "data/parking_video.mp4"
    CONFIG_PATH = "config/parking_spots.json"
    MODEL_NAME = "yolo11m.pt"
    INTERVAL = 2
    OVERLAP_THRESHOLD = 0.3  # 25% перекрытия
    
    # Проверка наличия файлов
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Ошибка: видеофайл не найден: {VIDEO_PATH}")
        print("   Поместите видео в папку data/parking_video.mp4")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Ошибка: конфигурация не найдена: {CONFIG_PATH}")
        print("   Сначала создайте config/parking_spots.json через marker.py")
        return
    
    # Создаём анализатор
    analyzer = VideoParkingAnalyzer(
        video_path=VIDEO_PATH,
        config_path=CONFIG_PATH,
        model_name=MODEL_NAME,
        interval=INTERVAL,
        save_video=True,
        overlap_threshold=OVERLAP_THRESHOLD
    )
    
    # Запускаем
    analyzer.run()


if __name__ == "__main__":
    main()