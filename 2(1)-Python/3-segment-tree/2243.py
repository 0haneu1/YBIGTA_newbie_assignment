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
    n = int(input().strip())
    
    # 맛의 범위가 최대 1,000,000이므로 세그먼트 트리 크기를 1,000,000으로 설정
    seg = SegmentTree[int, int](1_000_000)
    
    for _ in range(n):
        line = list(map(int, input().split()))
        A = line[0]
        if A == 1:
            # B번째로 맛있는 사탕 꺼내기
            B = line[1]
            # k번째 원소 찾기 → find_kth
            taste = seg.find_kth(cast(int, B))  
            print(taste)
            # 찾은 사탕 맛(taste)을 1개 제거
            seg.update_wrapper(taste, cast(int, -1))
        else:
            # A == 2, B = 맛, C = 개수(양수 = 추가, 음수 = 제거)
            B, C = line[1], line[2]
            seg.update_wrapper(B, cast(int, C))
    pass


if __name__ == "__main__":
    main()