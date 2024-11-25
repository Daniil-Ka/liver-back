import onnxruntime as ort
import numpy as np
import cv2

# Загрузите модель ONNX
model_path = "path/to/your/yolo_model.onnx"
session = ort.InferenceSession(model_path)

# Получаем имена входного и выходного тензоров
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Функция для обработки изображения
def preprocess_image(image_path, input_size=(640, 640)):
    image = cv2.imread(image_path)
    original_image = image.copy()
    # Изменение размера и нормализация
    resized_image = cv2.resize(image, input_size)
    blob = resized_image.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # Добавляем Batch-дименсию
    return blob, original_image

# Функция для визуализации результатов
def visualize_detections(image, detections, conf_threshold=0.5):
    for detection in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = detection
        if confidence > conf_threshold:
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, f"Class {int(class_id)}: {confidence:.2f}",
                        (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Подготовка изображения
image_path = "path/to/your/image.jpg"
input_data, original_image = preprocess_image(image_path)

# Запуск инференса
outputs = session.run([output_name], {input_name: input_data})[0]

# Постобработка результатов (адаптируйте под вашу модель)
# Обычно YOLO выводит: [x_min, y_min, x_max, y_max, confidence, class_id]
detections = []
for detection in outputs:
    x_min, y_min, x_max, y_max, conf, class_id = detection[:6]
    detections.append((x_min, y_min, x_max, y_max, conf, class_id))

# Визуализация
visualize_detections(original_image, detections)
cv2.imshow("Detections", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
