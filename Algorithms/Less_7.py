def two_sum(arr: list, k: int):
    """
    :param arr: list elements
    :param k: required sum
    :return: indices of two elements that add up to K
    """
    seen = {}
    for i, num in enumerate(arr):
        diff = k - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i


def k_sum_subarrays(arr: list, k: int) -> int:
    """
    :param arr: list elements
    :param k: required sum
    :return: count subarrays equalise k
    """
    seen = {0: 1}
    count = 0
    sum_ = 0
    for i in range(len(arr)):
        sum_ += arr[i]
        diff = sum_ - k

        if diff in seen:
            count += seen[diff]

        if sum_ in seen:
            seen[sum_] += 1
        else:
            seen[sum_] = 1

    return count


def solution():
    arr = list(map(int, input().split()))
    count = k_sum_subarrays(arr, 0)
    print(count)


if __name__ == '__main__':
    solution()
