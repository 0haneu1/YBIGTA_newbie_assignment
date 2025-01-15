from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)
        
        action: trie에 seq를 저장하기
        """
        node_idx = 0 
        for ch in seq:
            found_child = -1
            for cidx in self[node_idx].children:
                if self[cidx].body == ch:
                    found_child = cidx
 
            if found_child == -1:
                new_idx = len(self)
                self.append(TrieNode(body=ch))
                self[node_idx].children.append(new_idx)
                node_idx = new_idx
            else:
                node_idx = found_child
        self[node_idx].is_end = True

    def _ways(self, u: int, dp: dict[int, int], fact: list[int], MOD: int) -> int:
        """
        Calculate the number of possible alignment sequence of 'partial trie' which has root u.

        Returns: 
            - res (int): the number of possible alignment sequence
        """
        if u in dp:
            return dp[u]

        res = 1
        for nxt in self[u].children:
            res = (res * self._ways(nxt, dp, fact, MOD)) % MOD

        k = len(self[u].children)
        if self[u].is_end:
            k += 1

        res = (res * fact[k]) % MOD
        dp[u] = res
        return res

    def get_ways(self) -> int:
        """
        Calculates the number of possible alignment sequence of 'whole trie'
        in case which the sequencing rule is promised.
        """
        MOD = 1_000_000_007
        dp: dict[int, int] = {}
        MAX_CHILDS = 3000 
        fact = [1] * (MAX_CHILDS + 1)
        for i in range(1, MAX_CHILDS + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        dp = {}
        return self._ways(0, dp, fact, MOD) % MOD



import sys
from typing import Optional, Union

"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index: Optional[int] = None
        for c in trie[pointer].children:
            if trie[c].body == element:
                new_index = c
                break
        
        pointer = new_index if new_index is not None else pointer

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    input_data = sys.stdin.read().split()
    idx = 0
    while idx < len(input_data):
        N = int(input_data[idx])
        idx += 1
        words = input_data[idx:idx+N]
        idx += N

        trie: Trie[str] = Trie()
        for w in words:
            trie.push(w)

        total = 0
        for w in words:
            total += count(trie, w)

        print(f"{total / N:.2f}")


if __name__ == "__main__":
    main()