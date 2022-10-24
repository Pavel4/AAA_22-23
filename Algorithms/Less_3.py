def fib(n: int, a: int = 0, b: int = 1) -> int:
    if n <= 1:
        return a + b
    return fib(n - 1, b, a + b)


def get_subsets(original_set: list):
    if len(original_set) == 0:
        return [[]]

    all_subsets = [[]]
    for i, elem in enumerate(original_set):
        all_subsets += [[elem] + subset for subset in get_subsets(original_set[i + 1:])]

    return all_subsets


def has_subset_with_sum_k(arr, k, count=0, index=0, subarray=None):
    if subarray is None:
        subarray = []
    if index != len(arr):
        count = has_subset_with_sum_k(arr, k, count, index + 1, subarray)
        count = has_subset_with_sum_k(arr, k, count, index + 1,
                                      subarray + [arr[index]])
    else:
        if len(subarray) != 0 and sum(subarray) == k:
            count += 1

    return count


if __name__ == '__main__':
    pass
