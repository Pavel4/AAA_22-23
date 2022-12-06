def check_rotation(nums: list) -> int:
    len_ = len(nums)
    p = len_ // 2

    if p == 0:
        return nums[0]
    elif p == 1:
        return min(nums)

    if nums[p] > nums[0] > nums[len_ - 1]:
        return check_rotation(nums[p - 1:])
    else:
        return check_rotation(nums[:p + 1])


def find_elem_in_arr(arr: list, elem: int) -> int:
    l, r = 0, len(arr) - 1
    p = (l + r) // 2

    while l <= r:

        if elem == arr[p]:
            return p
        elif arr[l] <= arr[p]:
            if arr[l] <= elem <= arr[p]:
                r = p - 1
            else:
                l = p + 1
        else:
            if arr[p] <= elem <= arr[r]:
                l = p + 1
            else:
                r = p - 1

        p = (l + r) // 2

    return -1
