import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.geometry import rect_intersects_polygon


class VideoAnalyzer:
    def __init__(self, video_path, config_path='config/parking_spots.json', model_name='yolo11m-seg.pt', interval=2):
        self.video_path = video_path
        self.interval = interval
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.spots = self.config['spots']
        self.total_spots = len(self.spots)
        
        print("Загрузка модели YOLO11m...")
        self.model = YOLO(model_name)
        print("Модель загружена")
        
        self.cap = None
        self.connect_video()
    
    def connect_video(self):
        print(f"Открытие видео: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        print("Видео открыто")
    
    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.connect_video()
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("Видео закончилось, перемотка...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
            if not ret:
                raise RuntimeError("Не удалось перемотать видео")
        
        return frame
    
    def detect_cars(self, frame):
        results = self.model(frame, imgsz=1280)
        cars = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in [2, 3, 5, 7] and confidence > 0.5:
                        cars.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        return cars
    
    def analyze_frame(self, frame):
        cars = self.detect_cars(frame)
        print(f"Найдено транспорта: {len(cars)}")
        
        results = []
        for spot in self.spots:
            spot_id = spot['id']
            spot_polygon = spot['polygon']
            is_occupied = False
            
            for car in cars:
                if rect_intersects_polygon(car, spot_polygon):
                    is_occupied = True
                    break
            
            results.append({
                'id': spot_id,
                'occupied': is_occupied
            })
        
        total_free = sum(1 for r in results if not r['occupied'])
        
        return {
            'total_spots': self.total_spots,
            'free_spots': total_free,
            'occupied_spots': self.total_spots - total_free,
            'spots': results,
            'timestamp': time.time()
        }
    
    def visualize(self, frame, results):
        vis_frame = frame.copy()
        
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
        
        info = f"Свободно: {results['free_spots']} / {results['total_spots']}"
        cv2.putText(vis_frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(vis_frame, time_str, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return vis_frame
    
    def run(self):
        print("\n" + "=" * 60)
        print("ЗАПУСК АНАЛИЗА ВИДЕОФАЙЛА (YOLO11m)")
        print("=" * 60)
        print(f"Видео: {self.video_path}")
        print(f"Интервал: {self.interval} сек")
        print(f"Всего мест: {self.total_spots}")
        print("=" * 60)
        print("Нажмите 'q' для выхода\n")
        
        try:
            while True:
                start_time = time.time()
                
                frame = self.get_frame()
                results = self.analyze_frame(frame)
                
                print(f"\n{time.strftime('%H:%M:%S')} - Свободно: {results['free_spots']}/{results['total_spots']}")
                
                vis_frame = self.visualize(frame, results)
                cv2.imshow('Parking Analyzer (YOLO11m)', vis_frame)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nЗавершение работы...")
                    break
                
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Ресурсы освобождены")


def main():
    VIDEO_PATH = "data/parking_video.mp4"
    CONFIG_PATH = "config/parking_spots.json"
    INTERVAL = 2
    
    analyzer = VideoAnalyzer(
        video_path=VIDEO_PATH,
        config_path=CONFIG_PATH,
        model_name='yolo11m.pt',
        interval=INTERVAL
    )
    
    analyzer.run()


if __name__ == "__main__":
    main()
