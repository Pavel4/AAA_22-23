import heapq


def max_heapify(arr: list, i: int) -> None:
    left = i * 2 + 1
    right = i * 2 + 2

    if left >= len(arr):
        return

    if right < len(arr) and arr[left] < arr[right]:
        index_to_swap = right
        value_to_swap = arr[right]
    else:
        index_to_swap = left
        value_to_swap = arr[left]

    if arr[i] < value_to_swap:
        arr[i], arr[index_to_swap] = arr[index_to_swap], arr[i]
        max_heapify(arr, index_to_swap)


def build_max_heap(arr: list) -> None:
    for i in range(len(arr) // 2, -1, -1):
        max_heapify(arr, i)


def max_heapify_sized(arr: list, i: int, size: int) -> None:
    left = i * 2 + 1
    right = i * 2 + 2

    if left >= size:
        return

    if right < size and arr[left] < arr[right]:
        index_to_swap = right
        value_to_swap = arr[right]
    else:
        index_to_swap = left
        value_to_swap = arr[left]

    if arr[i] < value_to_swap:
        arr[i], arr[index_to_swap] = arr[index_to_swap], arr[i]
        max_heapify_sized(arr, index_to_swap, size)


def heapsort(arr: list) -> None:
    build_max_heap(arr)
    for i in range(len(arr) - 1, -1, -1):
        arr[0], arr[i] = arr[i], arr[0]
        max_heapify_sized(arr, 0, i)


def get_kth_element(arr: list, k: int) -> int:
    heapsort(arr)
    return arr[k]


def merge_k_sorted(arrs: list) -> list:

    h = []
    output = []

    for order, it in enumerate(map(iter, arrs)):
        next_ = it.__next__
        h.append([next_(), order, next_])

    heapq.heapify(h)

    while len(h) > 0:
        try:
            while True:
                value, order, next_ = s = h[0]
                output.append(value)
                s[0] = next_()
                heapq.heapreplace(h, s)
        except StopIteration:
            heapq.heappop(h)

    return output


def merge_k_sorted_without_iter(arrs: list) -> list:

    h = []
    output = []

    for order, arr in enumerate(arrs):
        if arr:
            elem, order, index = arr[0], order, 0
            h.append([elem, order, index])

    heapq.heapify(h)

    while len(h) > 0:
        elem, order, index = h[0]
        output.append(elem)
        if index + 1 != len(arrs[order]):
            elem_added = arrs[order][index + 1]
            heapq.heapreplace(h, [elem_added, order, index + 1])
        else:
            heapq.heappop(h)

    return output


if __name__ == '__main__':
    example_list = [4, 23, 54, 3, 22, 34, 33, 421, 0, 1, -1]
    print(get_kth_element(example_list, 4))
    print(merge_k_sorted_without_iter([[1, 2, 3], [3, 4, 5]]))
    print(merge_k_sorted([[1, 2, 3], [3, 4, 5]]))
