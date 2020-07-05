import numpy as np
import time
import random
import win32gui
import pyautogui
import operator
from itertools import permutations

#### Global variables ####

current_game_positions = np.zeros((12, 6), dtype=tuple)

##### Main patterns #####

### Crab, pufferfish, jellyfish ###

crab = np.array(['m']).reshape(1, 1)
jelly = np.array(['j']).reshape(1, 1)
puffer = np.array(['p']).reshape(1, 1)

images_location = "image_assets\\"

special_pieces = [crab, jelly, puffer]

#### 1 color patterns ####

### 1x3 ###
onetimesthree = np.array(['x', '?', 'x', 'x']).reshape(1, 4)

### 3x1 ###
threetimesone_1a = np.array(['?', 'x', 'x', '?', 'x', '?']).reshape(3, 2)

threetimesone_1b = np.array(['?', 'x', 'x', '?', '?', 'x']).reshape(3, 2)

### 4x1 (Good) ###
fourtimesone = np.array(
    ['x', '?', 'x', 'x', '?', 'x', '?', '?']).reshape(2, 4).T

### 5x1 (Great) ###
fivetimesone = np.array(['x', 'x', '?', 'x', 'x', '?',
                         '?', 'x', '?', '?']).reshape(2, 5).T

### 3x3 (Arrr!) ###
threetimesthree_1a = np.array(
    ['x', '?', 'x', 'x', '?', 'x', '?', '?', '?', 'x', '?', '?']).reshape(3, 4)

threetimesthree_1b = np.array(
    ['?', 'x', '?', '?', 'x', '?', 'x', 'x', '?', 'x', '?', '?']).reshape(3, 4)

### 3x4 (Arrr!) ###
threetimesfour = np.array(['?', 'x', '?', '?', 'x', '?', 'x',
                           'x', '?', 'x', '?', '?', '?', 'x', '?', '?']).reshape(4, 4)

### Ax5 (Yarrr!) ###

## 3x5 (Yarrr!) ##
threetimesfive = np.array(['?', 'x', '?', '?', '?', 'x', '?', '?', 'x',
                           '?', 'x', 'x', '?', 'x', '?', '?', '?', 'x', '?', '?']).reshape(5, 4)

onecolor_patterns_basic = [onetimesthree, threetimesone_1a, threetimesone_1b]

#### 2 color patterns ####

### 3x3 (Arrr!) ###
threetimesthree_2a = np.array(['x', 'y', 'x', 'y', 'y', 'x']).reshape(3, 2)

threetimesthree_2b = np.array(['x', 'y', 'y', 'x', 'x', 'y']).reshape(3, 2)

threetimesthree_2c = np.array(['x', 'x', 'y', 'x', 'y', 'y']).reshape(1, 6)

threetimesthree_2d = np.array(
    ['x', '?', 'x', 'y', 'y', 'x', '?', 'y']).reshape(4, 2)

threetimesthree_2e = np.array(
    ['x', '?', 'x', '?', 'y', 'x', '?', 'y', '?', 'y']).reshape(5, 2)

### 3x4 (Arrr!) ###
threetimesfour_2 = np.array(['x', '?', '?', '?', 'x', '?', '?',
                             '?', 'y', 'x', 'y', 'y', 'x', '?', '?', '?']).reshape(4, 4)

### 4x4 (Har!) ###
fourtimesfour_2 = np.array(
    ['x', 'y', 'x', 'y', 'y', 'x', 'x', 'y']).reshape(4, 2)

### Ax5 (Yarrr!) ###

## 3x5 (Yarrr!) ##
threetimesfive_2 = np.array(['x', '?', '?', '?', 'x', '?', '?', '?', 'y',
                             'x', 'y', 'y', 'x', '?', '?', '?', 'x', '?', '?', '?']).reshape(5, 4)

## 4x5 (Yarrr!) ##
fourtimesfive_2 = np.array(
    ['x', '?', 'x', 'y', 'y', 'x', 'x', 'y', 'x', 'y']).reshape(5, 2)

## 5x5 (Yarrr!) ##
fivetimesfive_2 = np.array(
    ['x', 'y', 'x', 'y', 'y', 'x', 'x', 'y', 'x', 'y']).reshape(5, 2)

twocolor_patterns_basic = [threetimesthree_2a, threetimesthree_2b, threetimesthree_2c, threetimesthree_2d,
                           threetimesthree_2e]

### AxBxC (Bingo) ###

## 3x3x3 (Bingo) ##
threecubed_a = np.array(['x', 'y', '?', '?', 'x', 'y',
                         '?', '?', 'y', 'x', 'y', 'y']).reshape(3, 4)

threecubed_b = np.array(['x', 'y', '?', '?', 'y', 'x',
                         'y', 'y', 'x', 'y', '?', '?']).reshape(3, 4)

## 3x3x4 (Bingo) ##
threesquaredfour_a = np.array(['x', 'y', '?', '?', 'x', 'y', '?',
                               '?', 'y', 'x', 'y', 'y', '?', 'y', '?', '?']).reshape(4, 4)

threesquaredfour_b = np.array(['x', 'y', '?', '?', 'x', 'y', '?',
                               '?', 'y', 'x', 'y', 'y', 'x', '?', '?', '?']).reshape(4, 4)

## 3x4x4 (Bingo) ##
threefoursquared = np.array(['x', 'y', '?', '?', 'x', 'y', '?',
                             '?', 'y', 'x', 'y', 'y', 'x', 'y', '?', '?']).reshape(4, 4)

## 3x3x5 (Bingo) ##
threesquaredfive_a = np.array(['x', 'y', '?', '?', 'x', 'y', '?', '?', 'y',
                               'x', 'y', 'y', '?', 'y', '?', '?', '?', 'y', '?', '?']).reshape(5, 4)

threesquaredfive_b = np.array(['x', 'y', '?', '?', 'x', 'y', '?', '?', 'y',
                               'x', 'y', 'y', 'x', '?', '?', '?', 'x', '?', '?', '?']).reshape(5, 4)

## 3x4x5 (Bingo) ##
threefourfive_a = np.array(['x', 'y', '?', '?', 'x', 'y', '?', '?', 'y',
                            'x', 'y', 'y', 'x', 'y', '?', '?', '?', 'y', '?', '?']).reshape(5, 4)

threefourfive_b = np.array(['x', 'y', '?', '?', 'x', 'y', '?', '?', 'y',
                            'x', 'y', 'y', 'x', 'y', '?', '?', 'x', '?', '?', '?']).reshape(5, 4)

## 3x5x5 (Bingo) ##
threefivesquared = np.array(['x', 'y', '?', '?', 'x', 'y', '?', '?', 'y',
                             'x', 'y', 'y', 'x', 'y', '?', '?', 'x', 'y', '?', '?']).reshape(5, 4)

### AxBxCxD (Sea Donkey!), A,B,C,D != 5 ###

## 3x3x3x3 (Sea Donkey!) ##
threequad_a = np.array(['x', 'x', 'y', 'x', 'y', 'y', '?', '?',
                        'x', 'y', '?', '?', '?', '?', 'x', 'y', '?', '?']).reshape(3, 6)

threequad_b = np.array(['?', '?', 'x', '?', '?', '?', 'x', 'x', 'y', 'x', 'y',
                        'y', '?', '?', 'x', 'y', '?', '?', '?', '?', '?', 'y', '?', '?']).reshape(4, 6)

threequad_c = np.array(['?', '?', 'x', 'y', '?', '?', 'x', 'x',
                        'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?']).reshape(3, 6)

threequad_d = np.array(['?', '?', 'x', '?', '?', '?', '?', '?', 'x', '?', '?', '?', 'x', 'x',
                        'y', 'x', 'y', 'y', '?', '?', '?', 'y', '?', '?', '?', '?', '?', 'y', '?', '?']).reshape(5, 6)

twocolor_patterns_best = [threecubed_a, threecubed_b,
                          threequad_a, threequad_b, threequad_c, threequad_d]

## 3x3x3x4 (Sea Donkey!) ##
threecubefour_a = np.array(['?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y', '?',
                            '?', 'x', 'x', 'y', 'x', 'y', 'y', '?', '?', 'x', '?', '?', '?']).reshape(4, 6)

threecubefour_b = np.array(['?', '?', 'x', '?', '?', '?', '?', '?', 'x', 'y', '?',
                            '?', 'x', 'x', 'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?']).reshape(4, 6)

threecubefour_c = np.array(['?', '?', 'x', '?', '?', '?', '?', '?', 'x', '?', '?', '?', 'x', 'x',
                            'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?', '?', '?', '?', 'y', '?', '?']).reshape(5, 6)

## 3x3x4x4 (Sea Donkey!) ##
threesquarefoursquare_a = np.array(['?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y',
                                    '?', '?', 'x', 'x', 'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?']).reshape(4, 6)

threesquarefoursquare_b = np.array(['?', '?', 'x', '?', '?', '?', '?', '?', 'x', 'y', '?', '?', 'x',
                                    'x', 'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?', '?', '?', '?', 'y', '?', '?']).reshape(5, 6)

### AxBxCx5 (Vegas) ###

## 3x3x3x5 (Vegas) ##
threecubefive_a = np.array(['?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y', '?', '?', 'x', 'x',
                            'y', 'x', 'y', 'y', '?', '?', 'x', '?', '?', '?', '?', '?', 'x', '?', '?', '?']).reshape(5, 6)

threecubefive_b = np.array(['?', '?', 'x', '?', '?', '?', '?', '?', 'x', 'y', '?', '?', 'x', 'x',
                            'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?', '?', '?', 'x', '?', '?', '?']).reshape(5, 6)

## 3x3x4x5 (Vegas) ##

threesquarefourfive = np.array(['?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y', '?', '?', 'x', 'x',
                                'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?', '?', '?', 'x', '?', '?', '?']).reshape(5, 6)

## 3x3x5x5 (Vegas) ##

threesquarefivesquare = np.array(['?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y', '?', '?', 'x',
                                  'x', 'y', 'x', 'y', 'y', '?', '?', 'x', 'y', '?', '?', '?', '?', 'x', 'y', '?', '?']).reshape(5, 6)
print("Basic 1 color patterns:")
for x in onecolor_patterns_basic:
    print(x)
print("Basic 2 color patterns:")
for x in twocolor_patterns_basic:
    print(x)
print("Best 2 color patterns:")
for x in twocolor_patterns_best:
    print(x)

##### Functions #####


def testing_function_array(array):
    for key in array:
        pyautogui.moveTo(key[0], key[1])
        pyautogui.click()
        time.sleep(1)


def testing_function(dict):
    for key in dict:
        pyautogui.moveTo(key[0], key[1])
        if dict[key] == 'a':
            print('blue_pentagon')
        elif dict[key] == 'b':
            print('darkblue_rectangle')
        elif dict[key] == 'c':
            print('green_square')
        elif dict[key] == 'd':
            print('green_octagon')
        elif dict[key] == 'e':
            print('lightblue_octagon')
        elif dict[key] == 'f':
            print('lightblue_sphere')
        elif dict[key] == 'g':
            print('teal_sphere')
        pyautogui.click()
        time.sleep(2)


def handle(title):
    return win32gui.FindWindowEx(0, 0, 0, title)


def focus_window(hwnd):
    win32gui.ShowWindow(hwnd, 9)
    win32gui.SetForegroundWindow(hwnd)
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    return rect[0], rect[1], width, height


def getclickpositions(matchindex, tuple1=(0, 0), tuple2=(0, 0)):

    print(current_game_positions)
    print(matchindex, tuple1, tuple2)

    if tuple1[0] == tuple2[0] and (abs(tuple1[1] - tuple2[1]) == 1 or abs(tuple1[1] - tuple2[1]) == 0):
        x = int((current_game_positions[tuple1[0] + matchindex[0], tuple1[1] + matchindex[1]]
                 [0] + current_game_positions[tuple2[0] + matchindex[0], tuple2[1] + matchindex[1]][0])/2)
        y = current_game_positions[tuple1[0] +
                                   matchindex[0]][tuple1[1] + matchindex[1]][1]
        return x, y
    else:
        raise ValueError('Check your tuple1, tuple2 values')


def piece_positions(xy):
    while True:
        location_a = pyautogui.locateAllOnScreen(f'{images_location}blue_pentagon.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_b = pyautogui.locateAllOnScreen(f'{images_location}darkblue_rectangle.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_c = pyautogui.locateAllOnScreen(f'{images_location}green_square.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_d = pyautogui.locateAllOnScreen(f'{images_location}green_octagon.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_e = pyautogui.locateAllOnScreen(f'{images_location}lightblue_octagon.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_f = pyautogui.locateAllOnScreen(f'{images_location}lightblue_sphere.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_g = pyautogui.locateAllOnScreen(f'{images_location}teal_sphere.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_m = pyautogui.locateAllOnScreen(f'{images_location}crab.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_j = pyautogui.locateAllOnScreen(f'{images_location}jelly.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)
        location_p = pyautogui.locateAllOnScreen(f'{images_location}puffer.png', region=(
            xy[0] + 92, xy[1] + 72, 270, 540), confidence=0.99)

        dict_init = {}

        for i in location_a:
            dict_init[i[0], i[1]] = 'a'
        for i in location_b:
            dict_init[i[0], i[1]] = 'b'
        for i in location_c:
            dict_init[i[0], i[1]] = 'c'
        for i in location_d:
            dict_init[i[0], i[1]] = 'd'
        for i in location_e:
            dict_init[i[0], i[1]] = 'e'
        for i in location_f:
            dict_init[i[0], i[1]] = 'f'
        for i in location_g:
            dict_init[i[0], i[1]] = 'g'
        for i in location_m:
            dict_init[i[0], i[1]] = 'm'
        for i in location_j:
            dict_init[i[0], i[1]] = 'j'
        for i in location_p:
            dict_init[i[0], i[1]] = 'p'

        if len(dict_init) == 72:
            break

    return dict_init


def readboardandsave(dict_init):
    positions = list(dict_init.keys())
    positions.sort(key=operator.itemgetter(1, 0))

    dict_2 = {}

    for position in positions:
        dict_2[position] = dict_init[position]

    i = 0

    for key in dict_2:
        positions[i] = list(key)
        i += 1

    i = 0
    j = 0

    column_positions = []
    row_positions = []

    while j < 12:
        row_positions.append(positions[j * 6][1])
        while i < 6:
            if j == 0:
                column_positions.append(positions[i][0])
            positions[j * 6 + i][1] = row_positions[j]
            positions[j * 6 + i][0] = column_positions[i]
            i += 1
        j += 1
        i = 0

    column_positions.sort()
    row_positions.sort()

    temp_array = list(dict_2.keys())

    i = 0

    for position in temp_array:
        temp_array[i] = list(position)
        i += 1

    i = 0

    for position in temp_array:
        while i < 6:
            if abs(position[0] - column_positions[i]) < 5:
                temp_array[temp_array.index(position)][0] = column_positions[i]
                break
            else:
                i += 1
        i = 0

    i = 0

    for position in temp_array:
        while i < 12:
            if abs(position[1] - row_positions[i]) < 5:
                temp_array[temp_array.index(position)][1] = row_positions[i]
                break
            else:
                i += 1
        i = 0

    i = 0

    for position in temp_array:
        temp_array[i] = tuple(position)
        i += 1

    values = list(dict_2.values())

    dict_3 = dict(zip(temp_array, values))

    positions = list(dict_3.keys())
    positions.sort(key=operator.itemgetter(1, 0))

    final_dict = {}

    for position in positions:
        final_dict[position] = dict_3[position]

    final_coordinates = np.zeros((12, 6), dtype=tuple)

    i = 0
    j = 0

    for position in list(final_dict.keys()):
        final_coordinates[j][i] = position
        i += 1
        if i % 6 == 0:
            i = 0
            j += 1

    global current_game_positions
    current_game_positions = final_coordinates

    return np.array(list(final_dict.values())).reshape(12, 6)


def pattern_variations(matrix):
    variation_matrix = []

    # teha deepcopy

    variation_matrix.append(matrix)

    if (matrix != np.fliplr(matrix)).any():
        variation_matrix.append(np.fliplr(matrix))
        if matrix.shape[0] != 1:
            variation_matrix.append(np.flipud(np.fliplr(matrix)))
    if (matrix != np.flipud(matrix)).any():
        variation_matrix.append(np.flipud(matrix))

    for variation in variation_matrix:
        yield variation


def piece_permutations():  # Generator object, must be iterated
    pieces = 'abcdefg'
    return permutations(pieces, 2)


def piece_type_combinations(matrix):
    pieces = 'abcdefg'

    temp_matrix = matrix.copy()

    # if temp_matrix == 1 color shit .. then use abcdefg..
    # if temp_matrix == 2 color... then use permutations

    i = 0

    if any(matrix[matrix == 'x']) == True and any(matrix[matrix == 'y']):
        indexes_x = np.where(matrix == 'x')
        indexes_y = np.where(matrix == 'y')
        for x in permutations(pieces, 2):
            while i < len(matrix[matrix == 'x']):
                temp_matrix[indexes_x[0][i]][indexes_x[1][i]] = x[0]
                i += 1
            i = 0
            while i < len(matrix[matrix == 'y']):
                temp_matrix[indexes_y[0][i]][indexes_y[1][i]] = x[1]
                i += 1
            i = 0
            yield temp_matrix

    elif any(matrix[matrix == 'x']) == True:
        indexes = np.where(matrix == 'x')
        for piece in pieces:
            while i < len(matrix[matrix == 'x']):
                temp_matrix[indexes[0][i]][indexes[1][i]] = piece
                i += 1
            i = 0
            yield temp_matrix


def generate2dmatrix_char(m, n, string):
    a = np.empty(m*n, dtype='str').reshape(m, n)
    for i in range(m):
        for j in range(n):
            a[i][j] = random.choice(string)
    return a


def comparearrays_eq(b, a):
    if len(a) == len(b):
        values = []*len(a)
        for i in range(len(a)):
            if a[i] == b[i] or (b[i] == '?' and a[i] != 'm' and a[i] != 'j' and a[i] != 'p'):
                values.append(True)
            else:
                values.append(False)
        return tuple(values)
    else:
        raise IndexError


def possiblearrangements(submatrix, matrix):
    return (sizeofmatrix(matrix)[1] - (sizeofmatrix(submatrix)[1] - 1)) * (sizeofmatrix(matrix)[0] - (sizeofmatrix(submatrix)[0] - 1))


def sizeofmatrix(matrix):
    if matrix.ndim >= 2:
        return matrix.shape
    elif matrix.ndim == 1:
        return matrix.shape[0], 1
    else:
        return 0


def issubmatrixnormal(submatrix, matrix):
    if submatrix.ndim != 0 and matrix.ndim != 0:
        if submatrix.ndim == matrix.ndim:
            if sizeofmatrix(submatrix) <= sizeofmatrix(matrix):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def findsubmatrix(submatrix, matrix):
    if issubmatrixnormal(submatrix, matrix):
        (M, N) = sizeofmatrix(matrix)
        (O, P) = sizeofmatrix(submatrix)
    else:
        return None

    i = 0
    k = 0
    matchindexes = [False]*possiblearrangements(submatrix, matrix)

    while i < M - (O - 1):
        j = 0
        while j < (N - P + 1):
            if all(comparearrays_eq(submatrix[0][:], matrix[i][0 + j: P + j])):
                if O > 1:
                    if all(comparearrays_eq(submatrix[1][:], matrix[i + 1][0 + j: P + j])):
                        if O > 2:
                            if all(comparearrays_eq(submatrix[2][:], matrix[i + 2][0 + j: P + j])):
                                if O > 3:
                                    if all(comparearrays_eq(submatrix[3][:], matrix[i + 3][0 + j: P + j])):
                                        if O > 4:
                                            if all(comparearrays_eq(submatrix[4][:], matrix[i + 4][0 + j: P + j])):
                                                matchindexes[k] = (i, j)
                                                k += 1
                                            else:
                                                if O == 4:
                                                    matchindexes[k] = (i, j)
                                                    k += 1
                                        else:
                                            matchindexes[k] = (i, j)
                                            k += 1
                                    else:
                                        if O == 3:
                                            matchindexes[k] = (i, j)
                                            k += 1
                                else:
                                    matchindexes[k] = (i, j)
                                    k += 1
                            else:
                                if O == 2:
                                    matchindexes[k] = (i, j)
                                    k += 1
                        else:

                            matchindexes[k] = (i, j)
                            k += 1
                    else:
                        if O == 1:
                            matchindexes[k] = (i, j)
                            k += 1
                else:
                    matchindexes[k] = (i, j)
                    k += 1
            else:
                pass
            j += 1
        i += 1
    if k > 0:
        return matchindexes[:k]
    else:
        return None


#'testboard.png - Paint'
HWND = handle('Puzzle Pirates - Vanhalgen on the Emerald ocean')
window_xy = focus_window(HWND)[:2]

A = readboardandsave(piece_positions(window_xy))

matcheszx = []
comboszx = []

start = time.time()
for pattern in onecolor_patterns_basic:
    #print('Pattern:\n', pattern)
    for variation in pattern_variations(pattern):
        #print('Variation:\n', variation)
        for combination in piece_type_combinations(variation):
            #print('Combinations:\n', combination)
            result = findsubmatrix(combination, A)
            if result != None:
                print('Result match:', result)
                matcheszx.append(result)
                comboszx.append(combination)
end = time.time()
print(comboszx)
print('Calculation took', end - start, 'seconds')
