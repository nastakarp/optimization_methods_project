import numpy as np
import sys
from numpy.linalg import norm
from numpy.linalg import inv
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

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


def goldensectionsearch(f, interval, tol):
    a = interval[0]
    b = interval[1]
    Phi = (1 + np.sqrt(5)) / 2
    L = b - a
    x1 = b - L / Phi
    x2 = a + L / Phi
    y1 = f(x1)
    y2 = f(x2)
    neval = 2
    xmin = x1
    fmin = y1

    # main loop
    while np.abs(L) > tol:
        if y1 > y2:
            a = x1
            xmin = x2
            fmin = y2
            x1 = x2
            y1 = y2
            L = b - a
            x2 = a + L / Phi
            y2 = f(x2)
            neval += 1
        else:
            b = x2
            xmin = x1
            fmin = y1
            x2 = x1
            y2 = y1
            L = b - a
            x1 = b - L / Phi
            y1 = f(x1)
            neval += 1

    answer_ = [xmin, fmin, neval]
    return answer_


def pparam(pU, pB, tau):
    if (tau <= 1):
        p = np.dot(tau, pU)
    else:
        p = pU + (tau - 1) * (pB - pU)
    return p


def doglegsearch(mod, g0, B0, Delta, tol):
    # dogleg local search
    xcv = np.dot(-g0.transpose(), g0) / np.dot(np.dot(g0.transpose(), B0), g0)
    pU = xcv * g0
    xcvb = inv(- B0)
    pB = np.dot(inv(- B0), g0)

    func = lambda x: mod(np.dot(x, pB))
    al = goldensectionsearch(func, [-Delta / norm(pB), Delta / norm(pB)], tol)[0]
    pB = al * pB
    func_pau = lambda x: mod(pparam(pU, pB, x))
    tau = goldensectionsearch(func_pau, [0, 2], tol)[0]
    pmin = pparam(pU, pB, tau)
    if norm(pmin) > Delta:
        pmin_dop = (Delta / norm(pmin))
        pmin = np.dot(pmin_dop, pmin)
    return pmin


def H(X, tol, df):
    X = X.flatten()
    n = len(X)
    ddf = np.zeros((n, n))
    delta = 0.1 * tol

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = delta
        df_plus = df(X + dx)
        df_minus = df(X - dx)
        ddf[:, i] = (df_plus - df_minus) / (2 * delta)
    return ddf


def trustreg(f, df, x0, tol):
    # TRUSTREG searches for minimum using trust region method
    # 	answer_ = trustreg(f, df, x0, tol)
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    # 	x0 - start point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords, radii]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics
    #   radii - array of trust regions radii

    delta = 1
    delta_max = 10
    eta = 0.1

    coordinates = [x0]
    xmin = x0
    radii = [delta]
    B = H(xmin, tol, df)
    m = lambda p: f(xmin) + np.dot(p.T, df(xmin)) + 0.5 * np.dot(np.dot(p.T, B), p)
    neval = 0

    while True:
        p = doglegsearch(m, df(xmin), B, delta, tol)
        rho = (f(xmin) - f(xmin + p)) / (m(np.zeros_like(p)) - m(p))

        if rho > eta:
            xmin = xmin + p

        if rho < 0.25:
            delta = delta / 4
        elif rho > 0.75 and norm(p) == delta:
            delta = min(2 * delta, delta_max)

        B = H(xmin, tol, df)

        coordinates.append(xmin)
        radii.append(delta)
        neval += 1

        if norm(df(xmin)) < tol or neval >= 1000:
            break

    fmin = f(xmin)

    answer_ = [xmin, fmin, neval, coordinates, radii]
    return answer_




def plotReg(x0, y0, Delta, ax):
    r = Delta
    color = [0, 0.4470, 0.7410]
    # Отрисовка круга
    circ = patches.Circle((x0, y0), radius=r, facecolor=color, ec='None', alpha=0.1)
    ax.add_patch(circ)


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
def trustregDraw(ax, coords, nsteps, radius):
    fSize = 11
    x0 = coords[0].flatten()
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i].flatten()
        x1 = coords[i + 1].flatten()
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=0.2)
        plotReg(x0[0], x0[1], radius[i], ax)

    ax.text(x1[0], x1[1] - 0.2, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)
    plotReg(x1[0], x1[1], radius[len(radius) - 1], ax)


def draw(coords, nsteps, flag, radius, f):
    fig, ax = plt.subplots()
    fig.suptitle('Trust region method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    trustregDraw(ax, coords, nsteps, radius)
    contourPlot(ax, f)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)



def main():
    x0 = np.array([[2.0], [1.0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fH, dfH, x0, tol)  # h - функция Химмельблау
    print(xmin, f, neval)
    draw(coords, len(coords), "h", rad, fH)

    print("Rosenbrock function:")
    x0 = np.array([[-2], [0]])
    tol = 1e-3
    [xmin, f, neval, coords, rad] = trustreg(fR, dfR, x0, tol)  # r - функция Розенброка
    print(xmin, f, neval)
    draw(coords, len(coords), "r", rad, fR)


if __name__ == '__main__':
    main()








