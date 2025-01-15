from lib import Trie
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