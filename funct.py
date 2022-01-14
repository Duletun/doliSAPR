import numpy
import matplotlib as ptl
from numpy import linalg

def generateReactionsMatrix(bars, left, right):
    count = len(bars)
    A = numpy.zeros((count+1, count+1), dtype=int).tolist()
    for i in range(count):
        A[i][i] += bars[i].A*bars[i].E/bars[i].L
        A[i][i+1] -= bars[i].A*bars[i].E/bars[i].L
        A[i+1][i] -= bars[i].A*bars[i].E/bars[i].L
        A[i+1][i+1] += bars[i].A*bars[i].E/bars[i].L
    if left == True:
        A[0][0] = 1
        A[1][0] = 0
        A[0][1] = 0
    if right == True:
        A[count][count] = 1
        A[count-1][count] = 0
        A[count][count-1] = 0
    return A

def generateReactionsGlobalVector(bars, F_to_points, left, right):
    count = len(bars)
    knots = [0] * (count+1)
    for conc in F_to_points:
        knots[int(conc.point)-1] = conc.power
    B = [0] * (count+1)
    for i in range(count+1):
        B[i] += knots[i]
        if i != 0:
            B[i] += bars[i-1].Q * bars[i-1].L/2
        if i != count:
            B[i] += bars[i].Q * bars[i].L/2
    if left == True:
        B[0] = 0
    if right == True:
        B[count] = 0
    return B

def generateDeltas(bars, F_to_points, left, right):
    count = len(bars)
    A = generateReactionsMatrix(bars, left, right)
    B = generateReactionsGlobalVector(bars, F_to_points, left, right)
    try:
        A = linalg.inv(A)
    except:
        linalg.lstsq(A, A)
    ans = numpy.dot(A,B)
    return ans

def solveN(bars, F_to_points, left, right):
    count = len(bars)
    A = generateReactionsMatrix(bars, left, right)
    B = generateReactionsGlobalVector(bars, F_to_points, left, right)
    try:
        A = linalg.inv(A)
    except:
        linalg.lstsq(A, A)
    v6 = numpy.dot(A,B)
    N  = numpy.zeros((count, 2), dtype=int).tolist()
    for i in range(count):
        N[i][0] = (bars[i].A * bars[i].E/bars[i].L) * (v6[i+1] -v6[i])
        if bars[i].Q != 0:
            N[i][0] += (bars[i].Q * bars[i].L / 2)
            N[i][1] -= bars[i].Q #* bars[i].L
    return N

def solveU(bars, F_to_points, left, right):
    count = len(bars)
    A = generateReactionsMatrix(bars, left, right)
    B = generateReactionsGlobalVector(bars, F_to_points, left, right)
    try:
        A = linalg.inv(A)
    except:
        linalg.lstsq(A, A)
    v6 = numpy.dot(A,B)
    U = numpy.zeros((count, 3), dtype=int).tolist()
    for i in range(count):
        U[i][0] = v6[i]
        U[i][1] = (v6[i+1] - v6[i]) / bars[i].L
        U[i][1] += (bars[i].Q * bars[i].L) / (2 * bars[i].E * bars[i].A)
        U[i][2] = -(bars[i].Q / (2*bars[i].E * bars[i].A))
    return U