"""11장 — data_drift/drift.ipynb 를 스크립트로 옮기고, 조건을 나눠 다시 잰다.

노트북은 "원본으로 학습한 모델"과 "드리프트를 넣은 데이터로 학습한 모델"의
아이템 임베딩을 deepchecks FeatureDrift로 비교한다. 여기서는 네 조건을 비교한다.

  A 노트북 그대로      : 드리프트 함수가 원본 DataFrame을 직접 고친다
  B 복사본으로 수정    : 원본과 드리프트 데이터를 분리한다
  C 대조군            : 같은 원본 데이터, 시드만 다르다 (드리프트 없음)
  D 시드 고정         : 원본과 드리프트 데이터, 시드는 같다 (데이터 차이만 남김)

사용법: ../.venv-ch11/bin/python drift_experiment.py
"""
import io
import os
import warnings
import zipfile

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset

DATA = os.path.expanduser("~/chapter-7-11-drill/data/ml-100k/u.data")
if not os.path.exists(DATA):
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://files.grouplens.org/datasets/movielens/ml-100k.zip").content))
    z.extractall(os.path.expanduser("~/chapter-7-11-drill/data"))


def load():
    df = pd.read_csv(DATA, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
    return df.iloc[:10000]


def introduce_user_drift(df, movie_ids, drift_amount=1):
    """노트북의 함수 그대로. df를 제자리에서 고치고 같은 객체를 돌려준다."""
    drift_indices = df["movieId"].isin(movie_ids)
    df.loc[drift_indices, "rating"] = df.loc[drift_indices, "rating"] + drift_amount
    df["rating"] = df["rating"].clip(1, 5)
    return df


class datasetReader(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df["userId"].values - 1).long()
        self.i = torch.tensor(df["movieId"].values - 1).long()
        self.r = torch.tensor(df["rating"].values).float()

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users + 1, n_factors)
        self.item_factors = torch.nn.Embedding(n_items + 1, n_factors)
        self.linear = torch.nn.Linear(n_factors, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()

    def forward(self, user, item):
        x = torch.mul(self.user_factors(user), self.item_factors(item))
        return self.linear2(self.dropout(self.relu(self.linear(x))))


def train_model(df, n_users, n_items, seed=None, epochs=5):
    """노트북과 같은 설정: SGD lr=1e-3, L1Loss, batch 64, 5 에폭."""
    if seed is not None:
        torch.manual_seed(seed)
    model = MatrixFactorization(n_users, n_items)
    init = model.item_factors.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    loss_func = torch.nn.L1Loss()
    loader = DataLoader(datasetReader(df), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for u, i, r in loader:
            loss = loss_func(model(u, i), r.unsqueeze(1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        sched.step()
    moved = (model.item_factors.weight.detach() - init).abs().mean().item()
    return model, moved


def drift_scores(m1, m2):
    from deepchecks.tabular import Dataset as DCDataset
    from deepchecks.tabular.checks import FeatureDrift

    e1 = m1.item_factors.weight.detach().numpy()
    e2 = m2.item_factors.weight.detach().numpy()
    cols = [f"item_factor_{i}" for i in range(e1.shape[0])]
    # 노트북과 같은 모양: 전치해서 열 하나가 아이템 하나, 행은 임베딩 차원 20개
    t1 = pd.DataFrame(e1.T, columns=cols)
    t2 = pd.DataFrame(e2.T, columns=cols)
    use = cols[:200]
    res = FeatureDrift(columns=use).run(
        DCDataset(t1[use], label=None, cat_features=[]), DCDataset(t2[use], label=None, cat_features=[])
    )
    scores = {c: v["Drift score"] for c, v in res.value.items()}
    method = res.value["item_factor_2"]["Method"]
    drifted = np.mean([scores[c] for c in cols[:100]])  # 드리프트를 넣은 영화 (movieId 1~100)
    other = np.mean([scores[c] for c in cols[100:200]])  # 넣지 않은 영화
    return scores["item_factor_2"], drifted, other, method, len(t1)


base = load()
n_users, n_items = int(base.userId.max()), int(base.movieId.max())
action_ids = list(range(1, 101))

# --- A: 노트북 그대로 ---
a_orig = load()
a_drift = introduce_user_drift(a_orig, action_ids)
print("A: 원본과 드리프트 데이터가 같은 객체인가:", a_orig is a_drift)
print("A: 두 데이터에서 값이 다른 평점 수:", int((a_orig["rating"].values != a_drift["rating"].values).sum()))

# --- B: 복사본 ---
b_orig = load()
b_drift = introduce_user_drift(b_orig.copy(), action_ids)
changed = int((b_orig["rating"].values != b_drift["rating"].values).sum())
print(f"B: 값이 다른 평점 수: {changed} / {len(b_orig)}")
print("   평균 평점:", round(b_orig.rating.mean(), 3), "->", round(b_drift.rating.mean(), 3))

rows = []
for name, d1, d2, s1, s2 in [
    ("A 노트북 그대로(같은 데이터, 시드 없음)", a_orig, a_drift, None, None),
    ("B 복사본으로 수정(시드 없음)", b_orig, b_drift, None, None),
    ("C 대조군(같은 원본, 시드 1과 2)", b_orig, b_orig, 1, 2),
    ("D 시드 고정(원본과 드리프트, 시드 1)", b_orig, b_drift, 1, 1),
]:
    m1, mv1 = train_model(d1, n_users, n_items, s1)
    m2, mv2 = train_model(d2, n_users, n_items, s2)
    f2, drifted, other, method, n = drift_scores(m1, m2)
    rows.append((name, f2, drifted, other, (mv1 + mv2) / 2))
    print(f"{name}: item_factor_2={f2:.3f} 드리프트영화평균={drifted:.3f} 그외평균={other:.3f}")

print("\n방법:", method, "/ 열 하나의 표본 수:", n)
print("학습으로 임베딩이 움직인 평균 크기(초깃값 표준편차는 1):", round(np.mean([r[4] for r in rows]), 5))
print("\n| 조건 | item_factor_2 | 드리프트 넣은 영화 100편 평균 | 넣지 않은 영화 100편 평균 |")
print("|---|---|---|---|")
for name, f2, drifted, other, _ in rows:
    print(f"| {name} | {f2:.3f} | {drifted:.3f} | {other:.3f} |")
