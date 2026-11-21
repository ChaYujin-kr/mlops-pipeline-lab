from bentoml.metrics import Counter

ranked_movie_present_counter = Counter(
    name="ranked_movie_present_counter",
    documentation="The number of times ranked movies is present in the request",
)
ranked_movie_absent_counter = Counter(
    # 변경: 원본은 두 카운터의 이름이 같아서 서버가 뜨지 않았다
    name="ranked_movie_absent_counter",
    documentation="The number of times ranked movies is absent in the request",
)
