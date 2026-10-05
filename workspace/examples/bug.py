"""故意包含数组越界 Bug，供后续代码审查演示。"""


def sum_numbers(numbers):
    total = 0
    # Bug：最后一次循环的索引等于列表长度，会触发 IndexError。
    for index in range(len(numbers) + 1):
        total += numbers[index]
    return total


if __name__ == "__main__":
    print(sum_numbers([1, 2, 3]))
