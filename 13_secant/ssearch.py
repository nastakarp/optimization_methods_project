import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

'''
def f(x):    return x**2 - 10*np.cos(0.3*np.pi*x) - 20

def df(x):    return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)
'''

#здесь нужно ставить интервал (6,9.9) иначе минимум не найдется
def f(x):
    return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def df(x):
    """
    Аналитическая производная f1(x).
    Не определена в x = -20 и x = 10 (особые точки из-за abs).
    Возвращает вещественное число; в особых точках — приближение.
    """
    eps = 1e-15  # порог для избежания деления на ноль

    # Часть 1: d/dx [ -2 * sin(sqrt(|x/2 + 10|)) ]
    u = x / 2 + 10
    abs_u = abs(u)
    if abs_u < eps:
        term1 = 0.0  # или np.nan — но 0 безопаснее для численных методов
    else:
        sign_u = np.sign(u)
        sqrt_abs_u = np.sqrt(abs_u)
        term1 = -2 * np.cos(sqrt_abs_u) * (sign_u / 4) / sqrt_abs_u
        # Обоснование: d/dx sqrt(|u|) = (1/(2*sqrt(|u|))) * sign(u) * du/dx,
        # где du/dx = 1/2 → итого: sign(u) / (4 * sqrt(|u|))

    # Часть 2: d/dx [ -x * sin(sqrt(|x - 10|)) ] = -sin(...) - x * cos(...) * d/dx[sqrt(|x-10|)]
    v = x - 10
    abs_v = abs(v)
    if abs_v < eps:
        # При v ≈ 0: sqrt(|v|) ≈ 0, sin(0)=0, cos(0)=1, но производная sqrt(|v|) → ∞
        # Однако множитель x * ... может компенсировать? Лучше использовать предел.
        # Практически: ставим 0 или игнорируем особую точку.
        term2 = -np.sin(0.0)  # = 0
        term3 = 0.0
    else:
        sign_v = np.sign(v)
        sqrt_abs_v = np.sqrt(abs_v)
        term2 = -np.sin(sqrt_abs_v)
        term3 = -x * np.cos(sqrt_abs_v) * (sign_v / (2 * sqrt_abs_v))

    return term1 + term2 + term3

#тут можно нормально ставить (-2,9.9)
def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df2(x):
    """
    Аналитическая производная f2(x) = x^2 - 10*cos(0.5*pi*x) - 110.
    f2'(x) = 2x + 5*pi*sin(0.5*pi*x)
    """
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)

def ssearch(interval, tol):
    a, b = interval
    dfa = df(a)
    dfb = df(b)
    neval = 2
    coords = []

    x = a - dfa * (b - a) / (dfb - dfa)
    dfx = df(x)
    neval += 1
    coords.append([x, a, b])

    while abs(dfx) > tol and abs(b - a) > tol:
        if dfx > 0:
            b = x
            dfb = dfx
        else:
            a = x
            dfa = dfx
        x = a - dfa * (b - a) / (dfb - dfa)
        dfx = df(x)
        neval += 1
        coords.append([x, a, b])

    xmin = x
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_


def drawdf(a, b, h, ax1):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b, h)
    y = [df(i) for i in x_]

    ax1.plot(x_, y, lw=1, c=color)
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)

    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")  # у вас стояло f''(x), но считаем же мы df

    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    color = (random.random(), random.random(), random.random())
    ax1.plot([a, b], [df(a), df(b)], marker='s', ms=3, c=color, lw=1)
    ax1.set_xlim([-2.0, 9.9])


def drawf(a, b, h, ax2):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b, h)
    y = [f(i) for i in x_]

    ax2.plot(x_, y, lw=1, c=color)
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)

    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")

    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax2.set_xlim([-2.0, 9.9])


def drawpointsDF(coord, ax1, i):
    ax1.plot(coord[0], df(coord[0]), marker='*')
    ax1.text(coord[0], df(coord[0]) + 3, str(i + 1))

    new_color = (random.random(), random.random(), random.random())
    ax1.scatter([coord[1], coord[2]], [df(coord[1]), df(coord[2])], marker='o', c=new_color)
    # secant
    ax1.plot([coord[1], coord[2]], [df(coord[1]), df(coord[2])], marker='s', ms=2, c=new_color, lw=1)
    ax1.plot([coord[0], coord[0]], [df(coord[0]), 0], marker='s', ms=2, c=new_color, lw=1)


def drawpointsF(coord, ax2, i):
    new_color = (random.random(), random.random(), random.random())
    ax2.plot(coord[0], f(coord[0]), marker='*')
    ax2.text(coord[0], f(coord[0]) + 1.4, str(i + 1))
    ax2.scatter([coord[1], coord[2]], [f(coord[1]), f(coord[2])], marker='o', c=new_color)


def secantsearchsecants(coords, interval):
    # draw graphics
    fig = plt.figure()
    # Для удобства анализа получившихся изображений можно настроить размер и пропорции выводимого окна - параметр figsize = [x, x] или width=\"XXX px\
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Secant search visualisation')

    a = interval[0]
    b = interval[1]
    h = (b - a) / 100

    drawdf(a, b, h, ax1)
    drawf(a, b, h, ax2)
    print("F(x) and F'(x):")
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

    DictName = []
    print("Steps:")
    nfigs = min([10, len(coords)])  # output 10 figures or less if number of points is less
    for i in range(nfigs):
        drawpointsDF(coords[i], ax1, i)
        drawpointsF(coords[i], ax2, i)

        ax1.set_title(str(i + 1) + " " + 'Iteration')
        DictName.append("iter" + str(i + 1) + ".png")
        fig.savefig(DictName[i])

    for elem in DictName:
        ad = "<img width=\"900px\" src=\"/resources/" + elem + "\">"
        print(ad)


def main():
    print("Find:")
    interval = [6, 9.9]
    tol = 1e-5
    [xmin, f, neval, coords] = ssearch(interval,tol)
    print([xmin, f, neval])
    secantsearchsecants(coords, interval)


if __name__ == '__main__':
    main()
