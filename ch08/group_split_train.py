"""8장 추가 실험 — 무작위 분할과 영상(클립) 단위 분할의 검증 점수 비교.

파일 이름 CA01_07.tif 에서 CA01 이 영상 하나이고, 영상마다 30장이다.
무작위로 나누면 같은 영상의 프레임이 학습과 검증에 함께 들어간다.

사용법: ../.venv-yolo/bin/python group_split_train.py <epochs>
"""
import glob
import os
import shutil
import sys
import time

import yaml
from sklearn.model_selection import GroupShuffleSplit
from ultralytics import YOLO

epochs = int(sys.argv[1])
SRC = os.path.expanduser("~/chapter-7-11-drill/data/DATASET/MINIDATA")
OUT = os.path.expanduser("~/chapter-7-11-drill/data/split_group")

images = sorted(glob.glob(os.path.join(SRC, "images", "**")))
labels = sorted(glob.glob(os.path.join(SRC, "labels", "**")))
groups = [os.path.basename(p).split("_")[0] for p in images]

train_idx, rest_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(images, groups=groups))
rest_groups = [groups[i] for i in rest_idx]
val_rel, test_rel = next(GroupShuffleSplit(1, test_size=0.4, random_state=42).split(rest_idx, groups=rest_groups))
splits = {"train": train_idx, "val": rest_idx[val_rel], "test": rest_idx[test_rel]}

shutil.rmtree(OUT, ignore_errors=True)
for name, idx in splits.items():
    for sub, files in [("images", images), ("labels", labels)]:
        os.makedirs(os.path.join(OUT, name, sub), exist_ok=True)
        for i in idx:
            shutil.copy2(files[i], os.path.join(OUT, name, sub))
    print(name, len(idx), "clips", sorted({groups[i] for i in idx}))

data_yaml = os.path.join(OUT, "data.yaml")
with open(data_yaml, "w") as f:
    yaml.dump(
        {**{k: os.path.join(OUT, k, "images") for k in splits}, "nc": 1, "names": {0: "id_card"}}, f
    )

t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(
    data=data_yaml, imgsz=640, epochs=epochs, batch=8, device=0, workers=2,
    project=os.path.expanduser("~/chapter-7-11-drill/ch08/runs"), name=f"group_{epochs}ep", exist_ok=True,
)
print("train_sec", round(time.time() - t0))
m = model.val(data=data_yaml, imgsz=640, batch=1, device=0, workers=2)
print("map50-95", round(m.box.map, 4), "map50", round(m.box.map50, 4), "map75", round(m.box.map75, 4))
