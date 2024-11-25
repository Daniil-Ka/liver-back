import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import io

from starlette.websockets import WebSocket

upload_router = APIRouter()

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from dicom2jpg import io2img
from PIL import Image

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
    Обрабатывает DICOM-файлы напрямую из байтового содержимого.
    """
    try:
        # Конвертируем DICOM в numpy.ndarray
        dicom_io = io.BytesIO(file_content)
        img_data = io2img(dicom_io)

        # Преобразуем numpy.ndarray в Pillow Image
        image = Image.fromarray(img_data)

        # Сохраняем обработанное изображение в память
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
    Обрабатывает изображения (JPEG, PNG).
    """
    try:
        image = Image.open(io.BytesIO(file_content))
        processed_image = image.convert("L")  # Пример обработки: перевод в градации серого

        image_io = io.BytesIO()
        processed_image.save(image_io, format="PNG")
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
    await websocket.accept()
    print("Клиент подключился.")
    t = 1
    try:
        while True:
            t += 1
            # Принимаем кадр (в формате Base64)
            data = await websocket.receive_text()

            # Извлекаем Base64-данные из строки
            if "," in data:
                base64_data = data.split(",")[1]
            else:
                base64_data = data

            # Декодируем Base64 в бинарный формат
            binary_data = base64.b64decode(base64_data)

            # Сохраняем кадр
            SAVE_DIR = Path("dir_test")
            SAVE_DIR.mkdir(exist_ok=True)  # Создаём директорию, если она не существует
            frame_path = SAVE_DIR / f"frame_.jpg"
            with open(frame_path, "wb") as f:
                f.write(binary_data)

            print(f"Кадр от клиента {websocket.client}: {frame_path}")

    except Exception as e:
        print("Ошибка WebSocket:", e)
    finally:
        print("Клиент отключился.")
