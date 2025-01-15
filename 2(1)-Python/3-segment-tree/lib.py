from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, List, cast


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    def __init__(self, size: int) -> None:
        """
        size: 관리할 배열(또는 값 범위)의 최대 크기
        """        
       
        self.size = size
        self.tree: List[T] = [cast(T, 0)] * (4 * size)

    def _update(self, idx: int, diff: T, node: int, start: int, end: int) -> None:
        """
        내부 재귀함수
        idx에 diff만큼 더해주고, 구간합 갱신
        node 가 [start, end] 구간을 관리한다고 가정
        """
        if idx < start or idx > end:
            return
        new_val = cast(int, self.tree[node]) + cast(int, diff)
        self.tree[node] = cast(T, new_val)

        if start == end:
            return
        
        mid = (start + end) // 2
        self._update(idx, diff, node * 2, start, mid)
        self._update(idx, diff, node * 2 + 1, mid + 1, end)

    def update_wrapper(self, idx: int, diff: T) -> None:
        """
        사용 예:
            - idx 위치의 값에 diff만큼 더한다 (예: 사탕 1개 추가/제거)
            - DVD 문제에서 idx 위치에 DVD를 '추가/제거' 한다고 볼 수도 있음
        """
        
        self._update(idx, diff, 1, 1, self.size)

    def _range_sum(self, left: int, right: int, node: int, start: int, end: int) -> T:
        """
        내부 재귀함수
        [left, right] 구간합을 구한다.
        """
        if right < start or left > end:
            # 구간이 겹치지 않으면, 0(T) 반환
            return cast(T, 0)
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        s1 = self._range_sum(left, right, node * 2, start, mid)
        s2 = self._range_sum(left, right, node * 2 + 1, mid + 1, end)
        new_val = cast(int, s1) + cast(int, s2)
        return cast(T, new_val)

    def sum_range(self, left: int, right: int) -> T:
        """
        [left, right] 구간합을 구하는 래퍼 함수
        """
        return self._range_sum(left, right, 1, 1, self.size)

    def _find_kth(self, k: U, node: int, start: int, end: int) -> int:
        """
        내부 재귀함수
        세그먼트 트리에 저장된 합(또는 카운트)을 이용해
        'k번째 원소(맛, 또는 위치)'의 인덱스를 찾는다
        
        - 가정: self.tree[node] >= k  (즉 현재 노드 구간에 최소 k개 이상 존재)
        - left 자식의 합이 k보다 크거나 같으면 왼쪽으로 이동
        - 작으면 오른쪽으로 이동하며, k에서 left 자식 합을 빼준다
        """
        if start == end:
            return start
        
        mid = (start + end) // 2
        left_sum = cast(int, self.tree[node * 2])

        if cast(int, k) <= left_sum:
            return self._find_kth(k, node * 2, start, mid)
        else:
            new_k = cast(int, k) - left_sum
            return self._find_kth(cast(U, new_k), node * 2 + 1, mid + 1, end)

    def find_kth(self, k: U) -> int:
        """
        1부터 size까지의 구간 중, 세그먼트 트리에 저장된
        'k번째 원소'가 위치한 인덱스를 반환.

        예) 사탕상자 문제에서 k번째로 맛있는 사탕 찾기
        예) DVD 문제에서는 보통 sum_range()를 쓰지만, 상황 따라 k-th가 필요할 수도 있음
        """
        return self._find_kth(k, 1, 1, self.size)