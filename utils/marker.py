import cv2
import json

# Список для хранения точек
points = []
spots = []

def click_event(event, x, y, flags, param):
    global points, spots
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Точка добавлена: ({x}, {y})")
        
        # Рисуем точку
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Разметка парковочных мест', img)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 4:
            spots.append({
                "id": len(spots) + 1,
                "polygon": points.copy()
            })
            print(f"Место {len(spots)} сохранено!")
            points = []

# Загружаем изображение
img = cv2.imread('data/test_image.jpg')
if img is None:
    print("❌ Ошибка: не удалось загрузить data/test_image.jpg")
    exit()
cv2.imshow('Разметка парковочных мест', img)
cv2.setMouseCallback('Разметка парковочных мест', click_event)

print("Инструкция:")
print("- Левый клик: добавить точку угла места")
print("- Правый клик: завершить ввод места (нужно минимум 4 точки)")
print("- Нажмите 'q' для сохранения и выхода")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Сохраняем разметку
with open('config/parking_spots.json', 'w', encoding='utf-8') as f:
    json.dump({"spots": spots}, f, indent=2, ensure_ascii=False)

print(f"Сохранено {len(spots)} парковочных мест")