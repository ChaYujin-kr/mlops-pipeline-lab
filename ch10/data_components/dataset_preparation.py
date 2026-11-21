from kfp import dsl
from kfp.dsl import Input, Output, Artifact


@dsl.component(base_image="python:3.11", packages_to_install=["requests"])
def download_ml25m_data(output_path_one: Output[Artifact], dataset: str = 'ml-25m'):
    import requests
    # 변경: 데이터셋 이름을 인자로 받는다 (ml-25m 262MB / ml-latest-small 1MB)
    url = f'https://files.grouplens.org/datasets/movielens/{dataset}.zip'
    response = requests.get(url, stream=True, verify=False, timeout=60)
    print(output_path_one.path)
    with open(output_path_one.path, 'wb') as file: 
        for chunk in response.iter_content(chunk_size=1024*1024):  # D
            if chunk:
                file.write(chunk)


@dsl.component(base_image="python:3.11")            
def unzip_data(input_path: Input[Artifact], ratings_output_path: Output[Artifact], movies_output_path: Output[Artifact], dataset: str = 'ml-25m'):
    import zipfile

    with zipfile.ZipFile(input_path.path, 'r') as z:
        with open(ratings_output_path.path, 'wb') as f:
            f.write(z.read(f'{dataset}/ratings.csv'))
        with open(movies_output_path.path, 'wb') as f:
            f.write(z.read(f'{dataset}/movies.csv'))


@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def split_dataset(input_parquet: Input[Artifact], dataset_path: Output[Artifact], random_state: int = 42):
    from sklearn.model_selection import train_test_split
    import os
    import pandas as pd
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    ratings_df = pd.read_parquet(input_parquet.path)

    # train is now 75% of the entire data set
    train, test = train_test_split(
        ratings_df,                                    
        test_size=1 - train_ratio,
        random_state=random_state)

    # 변경: 원본의 ratings_df.user_id.max() / item_id.max() 두 줄을 지웠다.
    # 열 이름은 userId, movieId라서 AttributeError가 나고, 결과도 쓰이지 않았다.

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    val, test = train_test_split(
        test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state)
    os.mkdir(dataset_path.path)
    train.to_parquet(os.path.join(dataset_path.path, 'train.parquet.gzip'), compression='gzip')
    test.to_parquet(os.path.join(dataset_path.path, 'test.parquet.gzip'), compression='gzip')
    val.to_parquet(os.path.join(dataset_path.path, 'val.parquet.gzip'), compression='gzip')


@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def csv_to_parquet(inputFile: Input[Artifact], output_path: Output[Artifact]):
    import pandas as pd
    df = pd.read_csv(inputFile.path, index_col=False)
    df.to_parquet(output_path.path, compression='gzip') 


@dsl.component(base_image="python:3.11", packages_to_install=["boto3"])
def put_to_minio(inputFile: Input[Artifact], upload_file_name:str='', bucket: str='datasets', dataset: str = 'ml-25m'):
    import boto3
    import os
    minio_client = boto3.client(                          
        's3',                                              
        endpoint_url='http://minio-service.kubeflow:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123')
    try:
        minio_client.create_bucket(Bucket=bucket)
    # 변경: 내가 만든 버킷이 이미 있으면 MinIO는 BucketAlreadyExists가 아니라
    # BucketAlreadyOwnedByYou를 돌려준다. 원본은 앞의 것만 잡아서 두 번째 실행부터 실패했다.
    except (minio_client.exceptions.BucketAlreadyExists, minio_client.exceptions.BucketAlreadyOwnedByYou):
        pass
    if os.path.isdir(inputFile.path):
        for file in os.listdir(inputFile.path):
            s3_path = os.path.join(dataset, file)
            minio_client.upload_file(os.path.join(inputFile.path, file), bucket, s3_path)
    else:
        if upload_file_name == '':
            _, file = os.path.split(inputFile.path)
        else:
            file = upload_file_name
        s3_path = os.path.join(dataset, file)
        minio_client.upload_file(inputFile.path, bucket, s3_path)


@dsl.component(base_image="python:3.11", packages_to_install=["pyarrow", "pandas"]) 
def qa_data(bucket: str = 'datasets', dataset: str = 'ml-25m', min_rows: int = 18750000):
    from pyarrow import fs, parquet
    print("Running QA")
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key='minio',
        secret_key='minio123',
        scheme='http')
    train_parquet = minio.open_input_file(f'{bucket}/{dataset}/train.parquet.gzip')
    df = parquet.read_table(train_parquet).to_pandas()
    assert df.shape[1] == 4
    # 변경: 최소 행 수를 인자로 받는다 (원본은 0.75 * 25e6 고정)
    print('rows', df.shape[0], 'cols', list(df.columns))
    assert df.shape[0] >= min_rows
    print('QA passed!')
