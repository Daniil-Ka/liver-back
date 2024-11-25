import base64
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageOps
from fastapi import APIRouter
from starlette.websockets import WebSocket
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from dicom2jpg import io2img
from PIL import Image
from model.model import model

upload_router = APIRouter()


@upload_router.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Обрабатывает загруженные файлы (DICOM или изображение).
    """
    try:
        file_content = await file.read()

        # Определяем, это DICOM или изображение
        if file.filename.endswith(".dcm"):
            return process_dicom(file_content)  # Обработка DICOM
        else:
            return process_image(file_content)  # Обработка изображения (JPEG/PNG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")


def process_dicom(file_content: bytes):
    """
    Обрабатывает DICOM-файлы напрямую из байтового содержимого,
    использует модель для анализа и возвращает обработанное изображение с масками объектов.
    """
    try:
        dicom_io = io.BytesIO(file_content)
        img_data = io2img(dicom_io)  # Конвертируем DICOM в numpy.ndarray

        # Проверка размерностей
        print(f"Размерность img_data: {img_data.shape}")
        if len(img_data.shape) > 3:
            raise ValueError("DICOM содержит больше измерений, чем поддерживается.")

        # Извлекаем первый слой, если это многослойный DICOM
        if len(img_data.shape) == 3:
            img_data = img_data[:, :, 0]

        # Нормализуем массив к диапазону 0-255 (если это необходимо)
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
        img_data = img_data.astype(np.uint8)

        # Преобразуем 2D массив в PIL Image
        image = Image.fromarray(img_data)

        # Масштабируем изображение до 640x640
        image = ImageOps.fit(image, (640, 640), Image.Resampling.LANCZOS)

        # Конвертируем изображение обратно в numpy.ndarray (RGB)
        img_data_rgb = np.array(image.convert("RGB"))

        # Прогон изображения через модель YOLO
        results = model.predict(img_data_rgb)

        # Создаём объект для рисования масок на изображении
        draw = ImageDraw.Draw(image)

        # Обработка масок
        if results[0].masks is not None:  # Проверяем наличие масок
            masks = results[0].masks.data.cpu().numpy()  # Маски (numpy array)
            for mask in masks:
                # Преобразуем каждую маску в бинарный массив
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))

                # Наносим полупрозрачную маску на изображение
                image.paste(Image.new("RGBA", image.size, (255, 0, 0, 100)), mask=mask_image)

        # Сохраняем обработанное изображение в поток
        image_io = io.BytesIO()
        image.save(image_io, format="JPEG")
        image_io.seek(0)

        return StreamingResponse(
            image_io,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=processed_dicom.jpg"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке DICOM-файла: {str(e)}")


def process_image(file_content: bytes):
    """
    Обрабатывает изображения.
    """
    try:
        # Открываем изображение
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Прогон через модель
        results = model.predict(np.array(image))

        # Нанесение результатов на изображение
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = f"{model.names[int(cls)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1), label, fill="blue")

        # Возвращаем обработанное изображение
        image_io = io.BytesIO()
        image.save(image_io, format="PNG")
        image_io.seek(0)

        return StreamingResponse(
            image_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=processed_image.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")


@upload_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """
    Обрабатывает данные через WebSocket (Base64 или DICOM).
    """
    await websocket.accept()
    print("Клиент подключился.")
    try:
        while True:
            # Получаем данные от клиента
            data = await websocket.receive_text()

            # Если данные в формате Base64
            if "," in data:
                base64_data = data.split(",")[1]
            else:
                base64_data = data

            # Декодируем данные
            binary_data = base64.b64decode(base64_data)

            # Проверяем, является ли это DICOM
            if binary_data.startswith(b"\x44\x49\x43\x4D"):  # Проверка на "DICM"
                response = process_dicom(binary_data)
            else:
                response = process_image(binary_data)

            # Сохраняем файл для отладки (опционально)
            SAVE_DIR = Path("dir_test")
            SAVE_DIR.mkdir(exist_ok=True)
            save_path = SAVE_DIR / "frame.jpg"
            with open(save_path, "wb") as f:
                f.write(binary_data)

            print(f"Файл обработан и сохранён: {save_path}")
    except Exception as e:
        print("Ошибка WebSocket:", e)
    finally:
        print("Клиент отключился.")