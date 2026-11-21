"""7장 실습 1 — 컴포넌트 사이에 데이터를 넘기는 두 가지 방법.

목표: 농담 N개를 파일로 만들고(아티팩트), 단어 수를 세서(파라미터) 출력한다.
참고 원본(먼저 보지 말고 막히면 보기):
  ~/chapter-8-9/object-detection-project/chapter_7/generate_jokes_pipeline.py

실행:
  source ~/chapter-7-drill/.venv-pipeline/bin/activate   # kfp 2.9.0
  python 01_joke_artifact.py                             # -> 01_joke_artifact.yaml
"""
from kfp import dsl
from kfp.dsl import Artifact, Input, Output


@dsl.component(base_image="python:3.11", packages_to_install=["pyjokes"])
def generate_joke(num_of_jokes: int, output_jokes: Output[Artifact]):
    import os

    import pyjokes

    os.makedirs(output_jokes.path, exist_ok=True)
    with open(os.path.join(output_jokes.path, "jokes.txt"), "w") as f:
        for _ in range(num_of_jokes):
            f.write(pyjokes.get_joke() + "\n")


@dsl.component(base_image="python:3.11")
def count_words(jokes_file: Input[Artifact]) -> int:
    # 이 함수는 별도 컨테이너에서 실행되므로 import는 함수 안에 둔다.
    import os

    with open(os.path.join(jokes_file.path, "jokes.txt")) as f:
        return len(f.read().split())


@dsl.component
def output_result(count: int) -> str:
    return f"Word count: {count}"


@dsl.pipeline(name="joke-artifact-pipeline")
def joke_pipeline(num_jokes: int = 3):
    jokes = generate_joke(num_of_jokes=num_jokes)
    # 아티팩트 출력은 이름으로 꺼내고(outputs["..."]), 반환값 하나는 .output으로 꺼낸다.
    count = count_words(jokes_file=jokes.outputs["output_jokes"])
    output_result(count=count.output)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=joke_pipeline, package_path="01_joke_artifact.yaml"
    )
