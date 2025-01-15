from lib import Trie
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