def knapsack(values: list, weights: list, capacity: int):
    dp_matrix = [[0 for _ in range(capacity + 1)] for _ in range(len(weights) + 1)]
    for i in range(len(weights)):
        for c in range(capacity + 1):
            if weights[i] <= c:
                dp_matrix[i + 1][c] = max(dp_matrix[i][c], values[i] + dp_matrix[i][c - weights[i]])
            else:
                dp_matrix[i + 1][c] = dp_matrix[i][c]

    return dp_matrix[-1][-1]


def solution_1():
    values = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    capacity = int(input())
    print(knapsack(values, weights, capacity))


def rob_corovans(values: list):
    optimal_heist = [0 for _ in range(len(values) + 3)]
    values = [0] * 3 + values
    for i in range(3, len(values)):
        optimal_heist[i] = max(optimal_heist[i - 1],
                               values[i] + max(optimal_heist[i - 2], optimal_heist[i - 3]))

    return optimal_heist[-1]


def solution_2():
    values = list(map(int, input().split()))
    print(rob_corovans(values))


if __name__ == '__main__':
    solution_2()
