from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import io

from starlette.websockets import WebSocket

upload_router = APIRouter()


@upload_router.post("/api/upload/")
async def upload_photo(file: UploadFile = File(...)):
    """
    Принимает фото от клиента, изменяет цвет и отправляет назад.
    """
    try:
        # Читаем содержимое файла
        file_content = await file.read()

        # Открываем изображение с помощью Pillow
        image = Image.open(io.BytesIO(file_content))

        # Пример изменения цвета: делаем изображение черно-белым
        processed_image = ImageOps.grayscale(image)

        # Сохраняем обработанное изображение в память
        image_io = io.BytesIO()
        processed_image.save(image_io, format="PNG")
        image_io.seek(0)  # Перемещаем курсор в начало для отправки клиенту

        # Возвращаем изображение назад клиенту
        return StreamingResponse(
            image_io, media_type="image/png", headers={"Content-Disposition": "inline; filename=processed_image.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@upload_router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Клиент подключился.")
    try:
        while True:
            # Принимаем кадр (в формате Base64)
            data = await websocket.receive_text()

            # Сохраняем кадр (опционально)
            frame_path = 'dir_test' / f"frame_{websocket.client}.jpg"
            with open(frame_path, "wb") as f:
                f.write(data.split(",")[1].encode("utf-8"))

            print(f"Кадр от клиента {websocket.client}: {frame_path}")

    except Exception as e:
        print("Ошибка WebSocket:", e)
    finally:
        print("Клиент отключился.")
