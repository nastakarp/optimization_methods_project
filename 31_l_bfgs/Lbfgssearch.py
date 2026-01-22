import numpy as np
import sys
from numpy.linalg import norm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# np.seterr(divide='ignore', invalid='ignore')


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v


def zoom(phi, dphi, alo, ahi, c1, c2):
    j = 1
    jmax = 1000
    while j < jmax:
        a = cinterp(phi, dphi, alo, ahi)
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a
        else:
            if abs(dphi(a)) <= -c2 * dphi(0):
                return a  # a is found
            if dphi(a) * (ahi - alo) >= 0:
                ahi = alo
            alo = a
        j += 1
    return a


def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        a = a0
        return a

    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        a = a0
        return a
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)

    return a


def wolfesearch(f, df, x0, p0, amax, c1, c2):
    a = amax
    aprev = 0
    phi = lambda x: f(x0 + x * p0)
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))

    phi0 = phi(0)
    dphi0 = dphi(0)
    i = 1
    imax = 1000
    while i < imax:
        if (phi(a) > phi0 + c1 * a * phi0) or ((phi(a) >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2)
            return a

        if abs(dphi(a)) <= -c2 * dphi0:
            return a  # a is found already

        if dphi(a) >= 0:
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a

        a = cinterp(phi, dphi, a, amax)
        i += 1

    return a


def addnew(arr, el):
    ln = 3
    if len(arr) < ln:
        arr.append(el)
    else:
        for i in range(len(arr)):
            if i < len(arr) - 1:
                arr[i] = arr[i + 1]
            else:
                arr[i] = el



def getz(df, d, y, l, x):
    q = df(x)
    m = []

    for i in range(len(l) - 1, -1, -1):
        alpha = l[i] * np.dot(d[i].T, q)
        m.append(alpha)
        q = q - alpha * y[i]

    H = np.dot(d[-1].T, y[-1]) / np.dot(y[-1].T, y[-1])
    z = H * q
    m = m[::-1]
    for i in range(len(l)):
        beta = l[i] * np.dot(y[i].T, z)
        z += d[i] * (m[i] - beta)
    return -z


def dfpsearch(f, df, x0, tol):
    # DFPSEARCH searches for minimum using DFP method
    # 	answer_ = dfpsearch(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics

    n = x0.size
    z = - df(x0)

    d = []
    y = []
    l = []

    coordinates = [x0]
    xmin = x0
    neval = 0

    while True:
        g = df(xmin)
        xl = xmin
        alpha = wolfesearch(f, df, xmin, z, 3, tol, 0.1)

        xmin = xmin + alpha * z

        coordinates.append(xmin)
        neval += 1

        addnew(d, xmin - xl)
        addnew(y, df(xmin) - df(xl))
        addnew(l, 1 / (np.dot(y[-1].T, d[-1])))

        z = getz(df, d, y, l, xmin)

        if norm(g) < tol or neval >= 1000:
            break

    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coordinates]
    return answer_

# def lbfgssearch(f, df, x0, tol):
#     m = 10
#     k = 0
#     kmax = 1000
#     coordinates = [x0]
#     dfx0 = df(x0)
#     z0 = -dfx0
#     neval = 1
#     D = []
#     Y = []
#     RO = []
#     while True:
#         a = wolfesearch(f, df, x0, z0, 3, tol, 0.1)
#         x1 = x0 + a*z0
#         d = x1 - x0
#         dfx1 = df(x1)
#         y = dfx1 - dfx0
#         neval += 1
#         ro = 1.0 / (np.dot(y.T, d))
#         RO.append(ro)
#         if ((norm(d) < tol) or (k >= kmax)):
#             xmin = x1
#             coordinates.append(x1)
#             break
#         else:
#             D.append(d)
#             Y.append(y)
#             if (len(D) > m):
#                 D.pop(0)
#                 Y.pop(0)
#                 RO.pop(0)
#             x0 = x1
#             dfx0 = dfx1
#             k+=1
#             coordinates.append(x0)
#             z0 = recursion(dfx0, D, Y, RO)
#     fmin = f(xmin)
#     answer_ = [xmin, fmin, neval, coordinates]
#     return answer_


# def recursion(g, D, Y, RO):
#     q = g
#     m = len(D)
#
#     AL = []
#
#     for i in range(m - 1, -1, -1):
#         alpha = RO[i] * np.dot(D[i].T, q)
#         AL.append(alpha)
#         q = q - alpha * Y[i]
#
#     H = np.dot(D[-1].T, Y[-1]) / np.dot(Y[-1].T, Y[-1])
#
#     z = H * q
#     AL = AL[::-1]
#     for i in range(m):
#         beta = RO[i] * np.dot(Y[i].T, z)
#         z += D[i] * (AL[i] - beta)
#     return -z




def contourPlot(ax, f):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)
    n = len(y1)

    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)

    # массивы для графиков функции и ее производных по x и y
    F = np.zeros((n, m))

    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def dfpDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0].flatten()
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i].flatten()
        x1 = coords[i + 1].flatten()
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)

    ax.text(x1[0], x1[1] - 0.4, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)


def draw(coords, nsteps, f, flag):
    fig, ax = plt.subplots()
    fig.suptitle('Davidon Fletcher Powell method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    dfpDraw(ax, coords, nsteps)
    contourPlot(ax, f)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)


def main():
    print("Rosenbrock function:")
    x0 = np.array([[-1], [-1]])
    tol = 1e-5
    [xmin, f, neval, coords] = dfpsearch(fR, dfR, x0, tol)  # функция Розенброка
    print(xmin, f, neval)
    draw(coords,  len(coords), fR, "r")


if __name__ == '__main__':
    main()
