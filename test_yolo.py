import cv2
import numpy as np
from ultralytics import YOLO
from utils.geometry import resize_mask_to_frame


def test_yolo_with_masks():
    model = YOLO('yolo11m-seg.pt')
    
    image_path = 'data/test_image.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    print(f"Изображение загружено: {image.shape}")
    
    results = model(image, imgsz=1280, retina_masks=True)
    
    print("\n" + "=" * 60)
    print("НАЙДЕННЫЕ ОБЪЕКТЫ (YOLO11m + Pixel-wise ROI):")
    print("=" * 60)
    
    found_vehicles = 0
    
    for r in results:
        boxes = r.boxes
        masks = r.masks
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                has_mask = False
                mask_info = ""
                if masks is not None and i < len(masks):
                    mask = masks[i].data[0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    mask_resized = resize_mask_to_frame(mask, image.shape)
                    pixel_count = np.sum(mask_resized)
                    has_mask = True
                    mask_info = f" | маска: {pixel_count} пикселей"
                
                print(f"  {class_name:15} | уверенность: {confidence:.3f} | "
                      f"координаты: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}){mask_info}")
                
                if class_id in [2, 3, 5, 7]:
                    found_vehicles += 1
    
    print("=" * 60)
    print(f"ВСЕГО объектов: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    print(f"ИЗ НИХ ТРАНСПОРТ: {found_vehicles}")
    print("=" * 60)
    
    if results[0].boxes is not None:
        annotated_image = results[0].plot()
        cv2.imwrite('data/yolo_result_pixelwise.jpg', annotated_image)
        print(f"\nИзображение с боксами сохранено в: data/yolo_result_pixelwise.jpg")
    else:
        print("\nМодель не нашла ни одного объекта на изображении")


if __name__ == "__main__":
    test_yolo_with_masks()
