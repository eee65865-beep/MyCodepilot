"""求和演示：首个数被遗漏，供 Agent 生成测试并发现错误。

输入契约：第一个整数 n，随后是 n 个整数（允许换行或空格分隔）。
0 <= n <= 10，每个数在 [-100, 100] 内。
正确输出：这 n 个整数的总和；n=0 时应输出 0。
"""

import sys


def sum_numbers(numbers):
    total = 0
    for index in range(1, len(numbers)):
        total += numbers[index]
    return total


def main():
    tokens = list(map(int, sys.stdin.read().split()))
    if not tokens:
        raise ValueError("Expected n followed by n integers")
    n, *numbers = tokens
    if not 0 <= n <= 10 or len(numbers) != n:
        raise ValueError("Expected 0 <= n <= 10 and exactly n integers")
    if any(not -100 <= number <= 100 for number in numbers):
        raise ValueError("Each integer must be in [-100, 100]")
    print(sum_numbers(numbers))


if __name__ == "__main__":
    main()
