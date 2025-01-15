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

        action: trie에 seq을 저장하기
        """
        node_idx = 0 
        
        for ch in seq:
            found = False
            for cidx in self[node_idx].children:
                if self[cidx].body == ch:
                    node_idx = cidx  
                    found = True
                    break
            
            if not found:
                new_node = TrieNode(body=ch)
                new_idx = len(self)
                self[node_idx].children.append(new_idx) 
                self.append(new_node)
                
                node_idx = new_idx 
        
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


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    lines: list[str] = sys.stdin.readlines()
    names = lines[1:]
    MOD = 1000000007

    trie: Trie[int] = Trie()
    result = 1

    for line in names:
        encoded = line.encode('ascii')  
        trie.push(encoded)             
    
    for node in trie:
        cnt = len(node.children) + node.is_end
        result *= factorial(cnt)
        result %= MOD
    
    print(result)


def factorial(n: int) -> int:
    result=1
    MOD = 1000000007
    for fac in range(1, n+1):
        result *= fac
        result %= MOD
    return result


if __name__ == "__main__":
    main()