# mlops-pipeline-lab

《Design a Machine Learning System (From Scratch)》 7~11장 실습 코드.
객체 탐지(YOLOv8)와 영화 추천 파이프라인을 노트북 한 대의 kind 클러스터(Kubeflow Pipelines, MinIO, MLflow)에서 돌렸다.

과정과 결과는 블로그 시리즈에 있다: [ML 파이프라인 실습 (7~11장)](https://chayujin-kr.github.io/posts/mlops-lab-00-overview/)

## 구성

| 폴더 | 내용 | 글 |
|---|---|---|
| `ch07/` | 농담 파이프라인(파라미터·아티팩트 전달), 탐지 데이터 준비 | [1편](https://chayujin-kr.github.io/posts/mlops-lab-01-kfp-data-passing/) |
| `ch08/` | YOLO 학습·검증 파이프라인(원본 / 수정본), 로컬 비교 실험 3개 | [2편](https://chayujin-kr.github.io/posts/mlops-lab-02-yolo-training-pipeline/) |
| `ch09/` | 탐지 모델 BentoML 서빙 | [3편](https://chayujin-kr.github.io/posts/mlops-lab-03-yolo-bentoml-serving/) |
| `ch10/` | 추천 데이터 준비, 학습·검증·등록 파이프라인 | [4편](https://chayujin-kr.github.io/posts/mlops-lab-04-recommender-pipeline/) |
| `ch11/` | 추천 서빙, 드리프트 대조 실험, 설명 가능성(EMF) | [5편](https://chayujin-kr.github.io/posts/mlops-lab-05-serving-drift-explain/) |
| `submit.py` | 컴파일된 파이프라인을 KFP에 제출하고 끝날 때까지 대기 | |

책 코드에서 바꾼 곳에는 `# 변경:` 주석을 달았다. `ch08/01_train_original.py`는 비교용으로 둔 원본이고 그대로는 실패한다.

## 실행

```bash
# 파이프라인 컴파일과 제출 (kfp 2.9.0)
kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80 &
python ch08/02_train_fixed.py
KFP_CACHE=1 python submit.py ch08/02_train_fixed.yaml train epochs=1 batch=2
```

데이터, 가중치(`*.pt`), 로그, 가상환경은 저장소에 넣지 않았다. 데이터는 파이프라인의 내려받기 단계가 받는다.
로컬 실험(`ch08/local_train.py` 등)은 미니 데이터셋을 `data/DATASET/`에 풀어 둔 상태를 전제로 한다.

MinIO 접속 정보 `minio` / `minio123`은 Kubeflow Pipelines 로컬 설치의 기본값이며, 책 코드에 있던 그대로다.

## 출처

파이프라인과 서빙 코드는 책의 예제 저장소를 바탕으로 수정한 것이다.

- `ch07/02_data_prep.py`, `ch08/01_*`, `ch08/02_*`, `ch09/` : [practical-mlops/object-detection-project](https://github.com/practical-mlops/object-detection-project)
- `ch10/`, `ch11/serving/`, `ch11/explainability/` : [practical-mlops/movie-recommender-project](https://github.com/practical-mlops/movie-recommender-project) (MIT License, Copyright (c) 2023 practical-mlops)

그 밖의 파일(`submit.py`, `ch07/01_joke_artifact.py`, `ch08/local_train.py`, `ch08/group_split_train.py`, `ch08/mispaired_experiment.py`, `ch11/drift_experiment.py`, `ch11/explainability/run_explain.py`)은 실습하면서 새로 쓴 것이다.
