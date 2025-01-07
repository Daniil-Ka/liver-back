import base64
from pathlib import Path
import numpy as np
from PIL import ImageDraw, ImageOps, ImageChops
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from dicom2jpg import io2img
from PIL import Image
from model.model import model

# Роутер для обработки загрузки файлов
upload_router = APIRouter()


@upload_router.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Обрабатывает загруженные файлы (DICOM или изображение).
    Принимает файл и в зависимости от его типа выполняет обработку.
    """
    try:
        file_content = await file.read()

        # Проверяем расширение файла и в зависимости от этого вызываем соответствующую обработку
        if file.filename.endswith(".dcm"):
            return process_dicom(file_content)  # Обработка DICOM
        else:
            return process_image(file_content)  # Обработка обычного изображения (JPEG/PNG)
    except Exception as e:
        # Возвращаем ошибку, если что-то пошло не так при обработке
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")


def process_dicom(file_content: bytes):
    """
    Обрабатывает DICOM-файлы.
    Преобразует DICOM в изображение, выполняет анализ с помощью модели и возвращает результат.
    """
    try:
        dicom_io = io.BytesIO(file_content)
        img_data = io2img(dicom_io)  # Конвертируем DICOM в numpy массив

        # Проверка размерности изображения и извлечение первого слоя (если многослойный)
        img_data = preprocess_image_data(img_data)

        # Преобразуем массив в изображение
        base_image = Image.fromarray(img_data).convert("L")

        # Масштабируем изображение до размера 640x640
        base_image = ImageOps.fit(base_image, (640, 640), Image.Resampling.LANCZOS)

        # Выполняем обработку изображения с помощью модели
        return process_image_common(base_image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке DICOM-файла: {str(e)}")


def process_image(file_content: bytes):
    """
    Обрабатывает обычные изображения (JPEG/PNG).
    """
    try:
        # Открываем изображение
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Масштабируем изображение до размера 640x640
        base_image = ImageOps.fit(image, (640, 640), Image.Resampling.LANCZOS)

        # Выполняем обработку изображения с помощью модели
        return process_image_common(base_image)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")


def preprocess_image_data(img_data: np.ndarray):
    """
    Общая предобработка изображения: нормализация, извлечение слоев.
    """
    if len(img_data.shape) > 3:
        raise ValueError("Изображение содержит больше измерений, чем поддерживается.")

    # Если изображение многослойное, извлекаем только первый слой
    if len(img_data.shape) == 3:
        img_data = img_data[:, :, 0]

    # Нормализуем изображение к диапазону 0-255
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
    return img_data.astype(np.uint8)


def process_image_common(base_image: Image):
    """
    Общая обработка изображения с наложением масок с использованием модели.
    """
    try:
        # Создаем RGBA-изображение для наложения маски
        overlay_image = base_image.convert("RGBA")

        # Прогоняем изображение через модель
        img_data_rgb = np.array(base_image)
        results = model.predict(img_data_rgb)

        # Если модель обнаружила маски, накладываем их на изображение
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = Image.new("L", overlay_image.size, 0)

            for mask in masks:
                mask_image = Image.fromarray((mask + 254).astype(np.uint8))
                combined_mask = ImageChops.lighter(combined_mask, mask_image)

            # Применяем маску на изображение
            overlay_image.putalpha(combined_mask)
        else:
            # Если масок нет, делаем фон прозрачным
            overlay_image.putalpha(254)

        # Конвертируем итоговое изображение в формат RGBA
        final_image = overlay_image.convert("RGBA")

        # Сохраняем обработанное изображение в поток
        image_io = io.BytesIO()
        final_image.save(image_io, format="PNG")
        image_io.seek(0)

        return StreamingResponse(
            image_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=processed_image.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")
