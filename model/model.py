# Инициализация модели YOLO
from pathlib import Path

from ultralytics import YOLO


path = Path(__file__).parent / "best (10).pt"
model = YOLO(path)
