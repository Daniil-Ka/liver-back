import time
from pathlib import Path
from ultralytics import YOLO

start_time = time.time()

path = Path(__file__).parent / "best (10).pt"
model = YOLO(path)

end_time = time.time()
loading_time = end_time - start_time

print(f'Модель загружена за {round(loading_time, 3)}с')
