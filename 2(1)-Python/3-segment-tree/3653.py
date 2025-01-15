from lib import SegmentTree
import sys
from typing import cast

"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    import sys
    input = sys.stdin.readline

    T = int(input().strip())
    for _ in range(T):
        n, m = map(int, input().split())
        # 볼 DVD 순서 (m개)
        watch_list = list(map(int, input().split()))

        # n번 DVD, 최대 m번 이동 → n + m 정도 범위를 커버하면 충돌 없이 맨 위를 새로 할당 가능
        seg = SegmentTree[int, int](n + m + 5)

        # pos[i]: 현재 DVD i의 위치
        pos = [0] * (n + 1)

        # 초기 상태: DVD i는 위치 i, 세그먼트 트리에 '1' 표시
        for i in range(1, n+1):
            pos[i] = i
            seg.update_wrapper(i, cast(int, 1))

        # "맨 위"로 올릴 때 사용할 offset
        offset = n
        
        results = []
        for dvd in watch_list:
            current_pos = pos[dvd]
            # dvd 위에 있는 개수 = seg.sum_range(1, current_pos - 1)
            above = seg.sum_range(1, current_pos - 1)
            results.append(str(above))

            # 현재 위치에서 DVD 제거
            seg.update_wrapper(current_pos, cast(int, -1))
            # 새로 맨 위로 올리기
            offset += 1
            pos[dvd] = offset
            seg.update_wrapper(offset, cast(int, 1))

        # 한 테스트 케이스의 결과 출력
        print(" ".join(results))
    pass


if __name__ == "__main__":
    main()