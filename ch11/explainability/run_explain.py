"""11장 — explainability/get_explanations.ipynb 를 스크립트로 옮긴 것.
사용법: ../../.venv-ch11/bin/python run_explain.py [expl_reg_term]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from data_reader.data_reader import DataReader
from explainer.emf import EMFExplainer
from model.emf import EMFModel
from recommender.recommender import Recommender

expl = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
np.random.seed(0)
DATA = os.path.expanduser("~/chapter-7-11-drill/data/ml-100k/u.data")
train_data = DataReader(DATA, sep="\t", names=["userId", "itemId", "rating", "timestamp"])

t0 = time.time()
emf = EMFModel(learning_rate=0.01, reg_term=0.001, expl_reg_term=expl, latent_dim=80, epochs=10, positive_threshold=3, knn=10)
emf.fit(train_data)
print("fit_sec", round(time.time() - t0))

t0 = time.time()
recommendations = Recommender(train_data, emf).recommend_all()
out = EMFExplainer(emf, recommendations, train_data).explain_recommendations()
print("recommend_explain_sec", round(time.time() - t0))

out["userId"] += 1
out["itemId"] += 1
pd.set_option("display.width", 200)
print(out.head(10).to_string())
print("rows", len(out), "columns", list(out.columns))
col = [c for c in out.columns if "expl" in c.lower()]
for c in col:
    v = out[c]
    print(c, "dtype", v.dtype)
    if v.dtype != object:
        print("  설명 값이 0보다 큰 추천 비율:", round(float((v > 0).mean()), 3), "평균:", round(float(v.mean()), 3))
out.to_csv("explanations_%s.csv" % expl, index=False)
