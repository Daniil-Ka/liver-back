from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import io

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
