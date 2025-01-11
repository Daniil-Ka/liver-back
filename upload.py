import base64
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageOps, ImageChops
from fastapi import APIRouter
from starlette.websockets import WebSocket, WebSocketDisconnect
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
    использует модель для анализа и возвращает обработанное изображение с цветными масками.
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

        # Преобразуем 2D массив в PIL Image (градации серого)
        base_image = Image.fromarray(img_data).convert("L")

        # Масштабируем изображение до 640x640
        base_image = ImageOps.fit(base_image, (640, 640), Image.Resampling.LANCZOS)

        # Создаём RGBA-изображение для наложения цветной маски
        overlay_image = base_image.convert("RGBA")

        # Прогон изображения через модель YOLO
        img_data_rgb = np.array(base_image.convert("RGB"))
        results = model.predict(img_data_rgb)

        # Наложение масок
        if results[0].masks is not None:  # Проверяем наличие масок
            masks = results[0].masks.data.cpu().numpy()  # Маски (numpy array)
            # Создаем пустую маску (изначально все черное, прозрачное)
            combined_mask = Image.new("L", overlay_image.size, 0)

            for mask in masks:
                mask_image = Image.fromarray((mask + 254).astype(np.uint8))

                # Объединяем текущую маску с общей маской
                combined_mask = ImageChops.lighter(combined_mask, mask_image)
            overlay_image.putalpha(combined_mask)
        else:
            overlay_image.putalpha(254)

        # Конвертируем обратно в RGB для сохранения
        final_image = overlay_image.convert("RGBA")

        # Сохраняем обработанное изображение в поток
        image_io = io.BytesIO()
        final_image.save(image_io, format="PNG")
        image_io.seek(0)

        return StreamingResponse(
            image_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=processed_dicom.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке DICOM-файла: {str(e)}")


def process_image(file_content: bytes):
    """
    Обрабатывает обычные изображения, использует модель для анализа и возвращает обработанное изображение с цветными масками.
    """
    try:
        # Открываем изображение
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Масштабируем изображение до 640x640
        base_image = ImageOps.fit(image, (640, 640), Image.Resampling.LANCZOS)

        # Прогон изображения через модель
        img_data_rgb = np.array(base_image)
        results = model.predict(img_data_rgb)  # Предсказание от модели

        # Создаем RGBA изображение для наложения маски
        overlay_image = base_image.convert("RGBA")

        # Наложение масок
        if results[0].masks is not None:  # Проверяем наличие масок
            masks = results[0].masks.data.cpu().numpy()  # Маски (numpy array)
            # Создаем пустую маску (изначально все черное, прозрачное)
            combined_mask = Image.new("L", overlay_image.size, 0)

            for mask in masks:
                mask_image = Image.fromarray((mask + 254).astype(np.uint8))

                # Объединяем текущую маску с общей маской
                combined_mask = ImageChops.lighter(combined_mask, mask_image)

            # Применяем комбинированную маску к изображению
            overlay_image.putalpha(combined_mask)
        else:
            overlay_image.putalpha(254)  # Полностью прозрачный фон

        # Конвертируем обратно в RGB для сохранения
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



@upload_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """
    Обрабатывает изображения через WebSocket.
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

            try:
                # Открываем изображение
                image = Image.open(io.BytesIO(binary_data)).convert("RGB")

                # Масштабируем изображение до 640x640
                base_image = ImageOps.fit(image, (640, 640), Image.Resampling.LANCZOS)

                # Прогон изображения через модель (если нужно)
                img_data_rgb = np.array(base_image)
                results = model.predict(img_data_rgb)  # Предсказание от модели

                # Создаем RGBA изображение для наложения маски
                overlay_image = base_image.convert("RGBA")

                # Наложение масок
                if results[0].masks is not None:  # Проверяем наличие масок
                    masks = results[0].masks.data.cpu().numpy()  # Маски (numpy array)
                    # Создаем пустую маску (изначально все черное, прозрачное)
                    combined_mask = Image.new("L", overlay_image.size, 0)

                    for mask in masks:
                        mask_image = Image.fromarray((mask + 254).astype(np.uint8))

                        # Объединяем текущую маску с общей маской
                        combined_mask = ImageChops.lighter(combined_mask, mask_image)

                    # Применяем комбинированную маску к изображению
                    overlay_image.putalpha(combined_mask)
                else:
                    overlay_image.putalpha(254)  # Полностью прозрачный фон

                # Конвертируем обратно в RGB для сохранения
                final_image = overlay_image.convert("RGBA")

                # Сохраняем обработанное изображение в поток
                image_io = io.BytesIO()
                final_image.save(image_io, format="PNG")
                image_io.seek(0)

                # Отправляем обработанное изображение через WebSocket
                await websocket.send_bytes(image_io.getvalue())
            except Exception as e:
                print(f"Ошибка обработки изображения: {e}")
                await websocket.send_text("Ошибка обработки изображения")

    except WebSocketDisconnect:
        print("Клиент отключился.")
    except Exception as e:
        print("Ошибка WebSocket:", e)
        await websocket.close()
