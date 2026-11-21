"""8장 추가 실험 — 책 원본 분할(이미지와 라벨이 어긋남)이 모델에 주는 영향.

원본 파이프라인이 실제로 만든 파일 목록(data/orig_lists)을 그대로 재현한다.
  mispaired : 학습 이미지 427장 + 그 실행의 labels 폴더 그대로 (짝이 맞는 것은 325장)
  correct   : 같은 학습 이미지 427장 + 올바른 라벨 427개
평가는 두 경우 모두 같은 143장(원본 실행의 검증+테스트 이미지)에 올바른 라벨을 붙여서 한다.

사용법: ../.venv-yolo/bin/python mispaired_experiment.py <epochs>
"""
import os
import shutil
import sys

import yaml
from ultralytics import YOLO

epochs = int(sys.argv[1])
ROOT = os.path.expanduser("~/chapter-7-11-drill/data")
SRC = os.path.join(ROOT, "DATASET/MINIDATA")
L = lambda name: [x.strip() for x in open(os.path.join(ROOT, "orig_lists", name)) if x.strip()]

train_images, train_labels = L("train_dataset.images.txt"), L("train_dataset.labels.txt")
eval_images = L("validation_dataset.images.txt") + L("test_dataset.images.txt")
stem = lambda f: os.path.splitext(f)[0]
paired = len({stem(f) for f in train_images} & {stem(f) for f in train_labels})
print("train images", len(train_images), "labels", len(train_labels), "paired", paired)


def build(name, images, labels):
    out = os.path.join(ROOT, "exp_" + name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images")); os.makedirs(os.path.join(out, "labels"))
    for f in images:
        shutil.copy2(os.path.join(SRC, "images", f), os.path.join(out, "images"))
    for f in labels:
        shutil.copy2(os.path.join(SRC, "labels", f), os.path.join(out, "labels"))
    return os.path.join(out, "images")


ev = build("eval", eval_images, [stem(f) + ".txt" for f in eval_images])
for name, labels in [("mispaired", train_labels), ("correct", [stem(f) + ".txt" for f in train_images])]:
    tr = build(name, train_images, labels)
    y = os.path.join(ROOT, f"exp_{name}.yaml")
    yaml.dump({"train": tr, "val": ev, "nc": 1, "names": {0: "id_card"}}, open(y, "w"))
    m = YOLO("yolov8n.pt")
    m.train(data=y, imgsz=640, epochs=epochs, batch=8, device=0, workers=2, seed=0, plots=False,
            project=os.path.join(os.path.dirname(ROOT), "ch08/runs"), name=f"exp_{name}", exist_ok=True, verbose=False)
    r = m.val(data=y, imgsz=640, batch=1, device=0, workers=2, plots=False, verbose=False)
    print(f"RESULT {name} precision={r.box.mp:.3f} recall={r.box.mr:.3f} map50={r.box.map50:.3f} map50-95={r.box.map:.3f}")
