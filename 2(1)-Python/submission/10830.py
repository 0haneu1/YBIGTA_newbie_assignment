from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None: # matrix 불러오는 초기화
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])]) 

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        self.matrix[key[0]][key[1]] = value
        pass

    def __matmul__(self, matrix: Matrix) -> Matrix:
        """
        Perform matrix multiplication, 'self @ matrix'.
        Calculate dot product of two matrix.
        if there are two matrix with size 'x X m'(self) and 'm1 X y'(matrix), 'm' should be equal to 'm1'

        Args:
            - matrix (Matrix): The right hand side matrix which will be multiplied with self.

        Returns:
            - Matrix: A new matrix resulting from the multiplication.

        Raises:
            - AssertionError: It is raised if m is not equal to m1.
        """
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += (self[i, k] * matrix[k, j]) % self.MOD
                    result[i, j] %= self.MOD

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        Compute the power of a matrix using exponentiation by squaring.

        This method calculates self raised to the power of n using a recursive
        divide-and-conquer approach.

        Args:
            n (int): The non-negative integer exponent.

        Returns:
            Matrix: A new matrix representing self raised to the power of n.

        Raises:
            AssertionError: If n is negative.
        """
        assert n >= 0

        if n == 0:
            return Matrix.eye(self.shape[0])
        if n == 1:
            return self

        half_pow = self ** (n // 2)
        result = half_pow @ half_pow

        if n % 2 == 1:
            result = result @ self

        return result


    def __repr__(self) -> str:
        """
        Return a string representation of the matrix.

        Returns:
            str: String representation of the matrix.
        """
        rows = []
        for i in range(self.shape[0]):
            row_str = " ".join(str(self[i, j] % self.MOD) for j in range(self.shape[1]))
            rows.append(row_str)
        return "\n".join(rows)


from typing import Callable
import sys


"""
아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()