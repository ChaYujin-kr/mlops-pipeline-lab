"""컴파일된 파이프라인 YAML을 KFP에 제출하고 끝날 때까지 기다린다.

사용법: python submit.py <pipeline.yaml> <실행 이름> [key=value ...]
전제:   kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80
"""
import sys
import time

import kfp

import os

CACHE = os.environ.get("KFP_CACHE") == "1"  # KFP_CACHE=1 이면 성공한 단계를 재사용
yaml_path, run_name, *kv = sys.argv[1:]
args = {}
for item in kv:
    k, v = item.split("=", 1)
    args[k] = int(v) if v.lstrip("-").isdigit() else v

client = kfp.Client(host="http://localhost:8080")
t0 = time.time()
run = client.create_run_from_pipeline_package(
    yaml_path,
    arguments=args,
    run_name=run_name,
    experiment_name="chapter-7-11",
    enable_caching=CACHE,
)
print("run_id", run.run_id, flush=True)
result = client.wait_for_run_completion(run.run_id, timeout=7200)
print("state", result.state, "elapsed_sec", round(time.time() - t0))
