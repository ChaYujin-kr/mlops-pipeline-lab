from kfp.dsl import component


@component(base_image="python:3.11", packages_to_install=["mlflow==2.17.2"])
def promote_model_to_staging(
        model_run_id: str,
        registered_model_name: str,
        rms_threshold: float,
        precision_threshold: float,
        top_k: int,
        recall_threshold: float,
        mlflow_uri: str):

    import mlflow.pytorch
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    mlflow.set_tracking_uri(uri=mlflow_uri)
    client = MlflowClient()

    current_staging = None
    try:
        current_staging = client.get_model_version_by_alias(registered_model_name, "staging")
    except RestException:
        print("No staging model found. Auto upgrade current run to staging.")

    # 변경: 첫 실행에는 staging 모델이 없어 current_staging이 None이다.
    # 원본은 None 검사 없이 .run_id를 읽어 AttributeError가 났다.
    if current_staging is not None and current_staging.run_id == model_run_id:
        print("Input run is already the current staging.")
        return

    if current_staging is not None:
        current_staging_model_data = client.get_run(current_staging.run_id).data.to_dictionary()
        staging_model_metrics = current_staging_model_data['metrics']

        new_model_data = client.get_run(model_run_id).data.to_dictionary()
        new_model_metrics = new_model_data['metrics']

        if (new_model_metrics['rms'] - staging_model_metrics['rms']) > rms_threshold:
            return

        if (new_model_metrics[f'precision_{top_k}'] - staging_model_metrics[f'precision_{top_k}']) < precision_threshold:
            return

        if (new_model_metrics[f'recall_{top_k}'] - staging_model_metrics[f'recall_{top_k}']) < recall_threshold:
            return

    # 변경: 이름을 하드코딩하지 않고 인자(registered_model_name)를 쓴다.
    result = mlflow.register_model(f"runs:/{model_run_id}/model", registered_model_name)
    client.set_registered_model_alias(registered_model_name, "staging", result.version)
    print(f"registered {registered_model_name} v{result.version} as staging")
