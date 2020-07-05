import numpy as np
import time
import cv2
from cv2 import matchTemplate as cv2m
import cProfile
from collections import deque

A = np.array([[0, 1, 1, 0, 1, 0],
              [0, 3, 4, 6, 0, 2],
              [1, 0, 4, 5, 0, 0],
              [0, 1, 0, 3, 3, 0],
              [0, 2, 4, 4, 0, 5],
              [2, 2, 3, 1, 0, 5],
              [3, 6, 5, 0, 5, 6],
              [1, 2, 0, 2, 0, 6],
              [0, 0, 1, 1, 0, 0],
              [2, 4, 0, 5, 5, 3],
              [1, 0, 4, 0, 2, 2],
              [3, 3, 1, 2, 6, 6]], dtype=object)


def match_in_fieldmatrix(boardstate, match):
    m = cv2m(boardstate.astype('uint8'), match.astype('uint8'), cv2.TM_SQDIFF)

    r, c = np.where(m == 0)

    if len(r) == 0:
        return None
    else:
        return list(zip(r, c))


def evaluate_combo(boardstate):
    for i in range(7):

        c = np.full((1, 3), i)

        if match_in_fieldmatrix(boardstate, c) or match_in_fieldmatrix(boardstate, c.T):
            return 3
    return 0


def elementswap_getchildren(matrix):

    height, width = matrix.shape

    for i, j in [(i, j) for i in range(height) for j in range(width - 1) if matrix[i, j] != matrix[i, j + 1]]:

        child = matrix.copy()

        child[i, j], child[i, j + 1] = child[i, j + 1], child[i, j]

        yield child


def bfs(initial, depth):
    visited = set()

    queue = deque([initial])

    i, j, k, toggle = 0, 0, 0, 0

    while queue:
        node = queue.popleft()

        node_tuple = tuple(map(tuple, node))

        if node_tuple not in visited:

            visited.add(node_tuple)

            # and visited[node_tuple] is None:
            if depth != 0 and evaluate_combo(node) == 0:
                for child in elementswap_getchildren(node):
                    queue.append(child)
                    i += 1

            if toggle == 0:

                k = i
                depth -= 1
                toggle = 1
        j += 1

        if j == k:
            k = i
            if depth != 0:
                depth -= 1

    return visited


if __name__ == "__main__":
    start = time.time()
    results = bfs(A, 4)
    end = time.time()

    print('Visited', len(results), 'positions')
    print('This took', round(end - start, 2), 'seconds')
