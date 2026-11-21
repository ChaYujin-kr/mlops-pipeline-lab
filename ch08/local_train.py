"""8장 비교용 — 파이프라인과 같은 분할(75/15/10, seed 42)로 로컬 GPU에서 학습한다.

사용법: ../.venv-yolo/bin/python local_train.py <epochs> <device: 0|cpu>
"""
import glob
import os
import shutil
import sys
import time

import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

epochs, device = int(sys.argv[1]), sys.argv[2]
SRC = os.path.expanduser("~/chapter-7-11-drill/data/DATASET/MINIDATA")
OUT = os.path.expanduser("~/chapter-7-11-drill/data/split")

# glob은 '.'으로 시작하는 파일(macOS의 ._ 파일)을 잡지 않는다. 정렬해서 쌍을 맞춘다.
images = sorted(glob.glob(os.path.join(SRC, "images", "**")))
labels = sorted(glob.glob(os.path.join(SRC, "labels", "**")))
assert [os.path.splitext(os.path.basename(p))[0] for p in images] == [
    os.path.splitext(os.path.basename(p))[0] for p in labels
]

x_train, x_rest, y_train, y_rest = train_test_split(images, labels, test_size=0.25, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.10 / 0.25, random_state=42)

if not os.path.exists(OUT):
    for name, xs, ys in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
        for sub, files in [("images", xs), ("labels", ys)]:
            os.makedirs(os.path.join(OUT, name, sub), exist_ok=True)
            for f in files:
                shutil.copy2(f, os.path.join(OUT, name, sub))
print("split", len(x_train), len(x_val), len(x_test))

data_yaml = os.path.join(OUT, "data.yaml")
with open(data_yaml, "w") as f:
    yaml.dump(
        {
            "train": os.path.join(OUT, "train", "images"),
            "val": os.path.join(OUT, "val", "images"),
            "test": os.path.join(OUT, "test", "images"),
            "nc": 1,
            "names": {0: "id_card"},
        },
        f,
    )

t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(
    data=data_yaml,
    imgsz=640,
    epochs=epochs,
    batch=8,
    device=device,
    workers=2,
    project=os.path.expanduser("~/chapter-7-11-drill/ch08/runs"),
    name=f"local_{device}_{epochs}ep",
    exist_ok=True,
)
print("train_sec", round(time.time() - t0))
m = model.val(data=data_yaml, imgsz=640, batch=1, device=device, workers=2)
print("map50-95", round(m.box.map, 4), "map50", round(m.box.map50, 4), "map75", round(m.box.map75, 4))
