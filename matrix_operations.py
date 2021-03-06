# -*- coding: utf-8 -*-
"""lin_reg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1scD3RDn2TmYpo2I4H52KfGjpBZBaqx1T

**РЕАЛИЗАЦИЯ МАТРИЧНЫХ ОПЕРАЦИЙ**

zeros(n, m) - матрица нулей

transp(m) - транспонирование

mult(m1, m2) - умножение

square_m(m) - проверка на квадратность (возвращает True, если квадратная)

copy_matrix(m) - копирует

determinant - находит определитель

det0_m - проверка на вырожденность (возвращает True или False)

equal_m - проверка на равенство

identity_matrix(n) - единичная

inv_m - обратная
"""

import math
import random

# матрица нулей
def zeros(rows, cols):
    matrix = []
    while len(matrix) < rows:
        matrix.append([])
        while len(matrix[-1]) < cols:
            matrix[-1].append(0.0)
    return matrix

# транспонирование матрицы
def transp(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transp_m = zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            transp_m[j][i] = matrix[i][j]

    return transp_m

#умножение матриц
def mult(m1,m2):
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])

    rows_m2 = len(m2)
    cols_m2 = len(m2[0])

    if cols_m1 != rows_m2:
        raise ArithmeticError('Number of m1 columns must equal number of m2 rows')

    mult = zeros(rows_m1, cols_m2)

    for i in range(rows_m1):
        for j in range(cols_m2):
            temp = 0
            for k in range(cols_m1):
                temp += m1[i][k] * m2[k][j]
            mult[i][j] = temp

    return mult

# проверка на квадратность
def square_m(matrix):
    if len(matrix) == len(matrix[0]):
        return True
    else:
        return False

#копирование матрицы
def copy_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    copy_of_m = zeros(rows, cols)

    for i in range(rows):
        for j in range(rows):
            copy_of_m[i][j] = matrix[i][j]
    return copy_of_m

#определитель
def determinant(matrix, eps=0):
    if len(matrix) == 2 and len(matrix[0]) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return det

    idxs = list(range(len(matrix)))

    for idx in idxs:
        temp_m = copy_matrix(matrix)
        temp_m = temp_m[1:]
        height = len(temp_m)

        for i in range(height):
            temp_m[i] = temp_m[i][0:idx] + temp_m[i][idx+1:]

        sign = (-1) ** (idx % 2)
        temp_det = determinant(temp_m)
        det += matrix[0][idx] * sign * temp_det

    return det

# проверка на вырожденность
def det0_m(matrix):
    det = determinant(matrix)
    if det != 0:
        return False
    else:
        return True

# проверка на равенство, мб eps не пригодится
def equal_m(m1,m2, eps=None):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return False

    for i in range(len(m1)):
        for j in range(len(m2[0])):
            if eps == None:
                if abs(m1[i][j] - m2[i][j]) > 1e-10:
                    return False
            else:
                if round(m1[i][j],eps) != round(m2[i][j],eps):
                    return False

    return True

#единичная матрица
def identity_matrix(n):
    I = zeros_matrix(n, n)
    for i in range(n):
        I[i][i] = 1.0

#обратная матрица
def inv_m(matrix, eps=None):
    if not square_m(matrix):
        print('matrix must be square')
        return

    if det0_m(matrix):
        print('matrix must be non singular')
        return

    size = len(matrix)
    m_copy = copy_matrix(matrix)
    I = identity_matrix(size)
    I_copy = copy_matrix(I)

    idxs = list(range(size))
    for k in range(size):
        fdScaler = 1.0 / m_copy[k][k]
        for j in range(size):
            m_copy[k][j] *= fdScaler
            I_copy[k][j] *= fdScaler
        for i in idxs[0:k] + idxs[k+1:]:
            crScaler = m_copy[i][k]
            for j in range(size):
                m_copy[i][j] = AM[i][j] - crScaler * m_copy[k][j]
                I_copy[i][j] = IM[i][j] - crScaler * I_copy[k][j]

    if equal_m(I,mult(matrix,I_copy),eps):
        return I_copy
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")
