import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


def f(x):
    return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def df(x):
    eps = 1e-15
    u = x / 2 + 10
    abs_u = abs(u)
    if abs_u < eps:
        term1 = 0.0
    else:
        sign_u = np.sign(u)
        sqrt_abs_u = np.sqrt(abs_u)
        term1 = -np.cos(sqrt_abs_u) * sign_u / (2 * sqrt_abs_u)

    v = x - 10
    abs_v = abs(v)
    if abs_v < eps:
        term2 = 0.0
        term3 = 0.0
    else:
        sign_v = np.sign(v)
        sqrt_abs_v = np.sqrt(abs_v)
        term2 = -np.sin(sqrt_abs_v)
        term3 = -x * np.cos(sqrt_abs_v) * sign_v / (2 * sqrt_abs_v)

    return term1 + term2 + term3

def ddf(x):
    """
    Вторая производная f1(x).
    Не определена в x = -20 и x = 10.
    Использует приближение вблизи особых точек.
    """
    eps = 1e-12

    # --- Часть 1: d²/dx² [ -2 * sin(sqrt(|x/2 + 10|)) ] ---
    u = x / 2 + 10
    abs_u = abs(u)
    if abs_u < eps:
        d2_term1 = 0.0
    else:
        sign_u = np.sign(u)
        sqrt_abs_u = np.sqrt(abs_u)
        cos_part = np.cos(sqrt_abs_u)
        sin_part = np.sin(sqrt_abs_u)

        # Производная от term1 = -cos(s) * sign(u) / (2*sqrt(|u|)), где s = sqrt(|u|)
        # d/dx [ -cos(s) * sign(u) / (2 s) ] =
        # = [ sin(s) * ds/dx * sign(u) / (2 s) ] + [ cos(s) * sign(u) / (4 s^3) * du/dx ]
        # где ds/dx = sign(u) / (4 s), du/dx = 1/2
        ds_dx = sign_u / (4 * sqrt_abs_u)
        du_dx = 0.5

        part_a = sin_part * ds_dx * sign_u / (2 * sqrt_abs_u)
        part_b = cos_part * sign_u * du_dx / (4 * (sqrt_abs_u ** 3))
        d2_term1 = part_a + part_b

    # --- Часть 2+3: d²/dx² [ -x * sin(sqrt(|x - 10|)) ] ---
    v = x - 10
    abs_v = abs(v)
    if abs_v < eps:
        d2_term2 = 0.0
    else:
        sign_v = np.sign(v)
        sqrt_abs_v = np.sqrt(abs_v)
        cos_v = np.cos(sqrt_abs_v)
        sin_v = np.sin(sqrt_abs_v)

        # Первая производная этой части: -sin(s) - x * cos(s) * sign(v) / (2 s)
        # Вторая производная:
        # d/dx [-sin(s)] = -cos(s) * ds/dx
        # d/dx [ -x * cos(s) * sign(v) / (2 s) ] =
        #   = -cos(s)*sign(v)/(2s)
        #     - x * [ -sin(s)*ds/dx * sign(v)/(2s) + cos(s)*sign(v)/(4 s^3) * dv/dx ]
        #
        # где s = sqrt(|v|), ds/dx = sign(v) / (2 * 2 s) = sign(v) / (4 s), dv/dx = 1

        ds_dx = sign_v / (4 * sqrt_abs_v)

        # Производная от -sin(s)
        d2a = -cos_v * ds_dx

        # Производная от -x * cos(s) * sign(v) / (2 s)
        term_base = -cos_v * sign_v / (2 * sqrt_abs_v)
        derivative_of_product = (
                term_base +
                x * sin_v * ds_dx * sign_v / (2 * sqrt_abs_v) -
                x * cos_v * sign_v / (4 * (sqrt_abs_v ** 3))
        )
        d2_term2 = d2a + derivative_of_product

    return d2_term1 + d2_term2

def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df2(x):
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)

def ddf2(x):
    """
    Вторая производная f2(x) = x^2 - 10*cos(0.5*pi*x) - 110
    f2'(x) = 2x + 5π * sin(0.5π x)
    f2''(x) = 2 + 5π * cos(0.5π x) * (0.5π)
            = 2 + (5π²/2) * cos(0.5π x)
    """
    return 2 + (5 * np.pi ** 2 / 2) * np.cos(0.5 * np.pi * x)

'''
def f(x): return x**2 - 10*np.cos(0.3*np.pi*x) - 20

def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def ddf(x): return 2 + 0.9*(np.pi**2)*np.cos(0.3*np.pi*x)
'''


def nsearch(tol, x0):
    # NSEARCH searches for minimum using Newton method
    # 	answer_ = nsearch(tol,x0)
    #   INPUT ARGUMENTS
    # 	tol - set for bot range and function value
    #	x0 - starting point
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of x values found during optimization

    x = x0
    coords = []
    neval = 0
    coords.append(x)

    while np.abs(df(x)) > tol:
        neval += 3
        x = x - df(x) / ddf(x)
        coords.append(x)
    xmin = x
    fmin = f(x)
    answer_ = [xmin, fmin, neval, coords]
    return answer_


def drawdf(a, b, h, ax1):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [df(i) for i in x_]

    ax1.plot(x_, y, lw=1, c=color)
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)

    ax1.set_xlabel('x')
    ax1.set_ylabel("f'(x)")

    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax1.set_xlim([-2.1, 10.1])
    ax1.set_ylim([-30, 40])


def drawf(a, b, h, ax2):
    color = (random.random(), random.random(), random.random())
    x_ = np.arange(a, b + h, h)
    y = [f(i) for i in x_]

    ax2.plot(x_, y, lw=1, c=color)
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)

    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")

    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)
    ax2.set_xlim([-2.1, 10.1])


def drawpointsDF(coord, ax1, a, b, h, i):
    ax1.plot(coord, df(coord), marker='*')
    ax1.text(coord, df(coord) + 4, str(i + 1), backgroundcolor='white')
    # tangent
    x_ = np.arange(a, b + h, h)
    new_color = (random.random(), random.random(), random.random())
    k = ddf(coord)
    br = df(coord)
    yt = k * (x_ - coord) + br
    ax1.plot(x_, yt, c=new_color, lw=1)
    ax1.plot([coord, coord], [0, df(coord)], c=(0, 0, 0), lw=0.5)


def drawpointsF(coord, ax2, i):
    ax2.plot(coord, f(coord), marker='*')
    ax2.text(coord, f(coord) + 4, str(i + 1), backgroundcolor='white')


def newtondrawfig(interval, coord):
    # draw graphics
    fig = plt.figure(figsize=[20, 10])
    # Для удобства анализа получившихся изображений можно настроить размер и пропорции выводимого окна - параметр figsize = [x, x] или width=\"XXX px\
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Newton visualisation')

    a = interval[0]
    b = interval[1]
    h = (b - a) / 100

    # plot of df function

    drawdf(a, b, h, ax1)
    drawf(a, b, h, ax2)
    print("F(x) and F'(x):")
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"
    print(ad)

    DictName = []
    print("Steps:")
    nfigs = min([10, len(coord)])  # output 10 figures or less if number of points is less
    for i in range(nfigs):
        drawpointsDF(coord[i], ax1, a, b, h, i)
        drawpointsF(coord[i], ax2, i)

        ax1.set_title(str(i + 1) + " " + 'Iteration')
        DictName.append("iter" + str(i + 1) + ".png")
        fig.savefig(DictName[i])

    for elem in DictName:
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"
        print(ad)


def main():
    print("Find:")
    interval = [-2, 10] #for drawing
    tol = 1e-5
    [xmin, f, neval, coords] = nsearch(tol, 9.5) #МЕНЯЙ Х0 1-9.5, 2-0.5
    print([xmin, f, neval])
    newtondrawfig(interval, coords)


if __name__ == '__main__':
    main()
