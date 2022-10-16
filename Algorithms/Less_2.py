# 1
def validate_pushed_popped(pushed: list, popped: list) -> bool:
    stack = []
    p1, p2 = 0, 0
    while p1 != len(pushed):
        stack.append(pushed[p1])
        p1 += 1
        while stack and stack[-1] == popped[p2]:
            stack.pop()
            p2 += 1
    return stack == popped[p2:][::-1]


def solution_1():
    pushed = list(map(int, input().split()))
    popped = list(map(int, input().split()))
    result = validate_pushed_popped(pushed, popped)
    print(result)


# 2
def calculate_stock_spans(prices: list) -> list:
    len_prices = len(prices)
    answer = [1] * len_prices
    stack = [0]
    for i in range(1, len_prices):
        while stack and prices[i] >= prices[stack[-1]]:
            stack.pop()
        answer[i] = i - stack[-1] if stack else i + 1
        stack.append(i)

    return answer


def solution_2():
    prices = list(map(int, input().split()))
    # prices = [100, 10, 40, 50, 60, 100, 90]
    spans = calculate_stock_spans(prices)
    print(' '.join(map(str, spans)))


if __name__ == '__main__':
    solution_1()
    solution_2()
