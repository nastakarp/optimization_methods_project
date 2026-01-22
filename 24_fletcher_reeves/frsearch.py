from numpy.linalg import norm  # Импортирует функцию norm для вычисления евклидовой нормы вектора.
import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI.
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt.

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fH(X):
    x = X[0]  # Извлекает первую компоненту вектора X (x).
    y = X[1]  # Извлекает вторую компоненту (y).
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Вычисляет значение функции Химмельблау.
    return v  # Возвращает скалярное значение функции.

# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)  # Создаёт копию входного вектора для хранения градиента.
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)  # Частная производная по x.
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)  # Частная производная по y.
    return v  # Возвращает градиент как вектор [df/dx, df/dy].

# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # Функция Розенброка ("банановая долина").
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
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)  # ∂f/∂x.
    v[1] = 200 * (y - x ** 2)  # ∂f/∂y.
    return v

def zoom(phi, dphi, alo, ahi, c1, c2):
    # Фаза "zoom" алгоритма линейного поиска по условиям Вульфа.
    j = 1  # Счётчик итераций внутри zoom.
    jmax = 1000  # Максимальное число итераций.
    while j < jmax:
        a = cinterp(phi, dphi, alo, ahi)  # Интерполяция для нахождения нового кандидата a.
        # Проверка условия достаточного убывания (Армихо):
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a  # Сужаем правую границу.
        else:
            # Проверка условия кривизны:
            if abs(dphi(a)) <= -c2 * dphi(0):
                return a  # Найден подходящий шаг.
            # Если производная положительна — минимум между alo и a:
            if dphi(a) * (ahi - alo) >= 0:
                ahi = alo
            alo = a  # Сужаем левую границу.
        j += 1
    return a  # Возвращаем последнее найденное значение (аварийный выход).

def cinterp(phi, dphi, a0, a1):
    # Кубическая интерполяция для нахождения минимума одномерной функции phi(a).
    # Защита от деления на ноль и NaN:
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
    # Линейный поиск с выполнением условий Вульфа.
    a = amax  # Начальное значение шага.
    aprev = 0  # Предыдущее значение шага.
    # Определяем одномерные функции phi(a) = f(x0 + a*p0) и её производную:
    phi = lambda x: f(x0 + x * p0)
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))

    phi0 = phi(0)  # Значение функции в нуле.
    dphi0 = dphi(0)  # Производная в нуле.
    i = 1
    imax = 1000
    while i < imax:
        # Проверка условия Армихо или того, что функция начала расти:
        if (phi(a) > phi0 + c1 * a * dphi0) or ((phi(a) >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2)  # Переход в фазу zoom.
            return a

        # Проверка условия кривизны:
        if abs(dphi(a)) <= -c2 * dphi0:
            return a  # Шаг найден.

        # Если производная положительна — минимум левее:
        if dphi(a) >= 0:
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a

        # Иначе расширяем интервал:
        a = cinterp(phi, dphi, a, amax)
        i += 1

    return a  # Аварийный выход.

def prsearch(f, df, x0, tol):
    # PRSEARCH searches for minimum using Polak-Ribiere method
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]

    coordinates = [x0]  # Хранит траекторию поиска.
    xmin = x0  # Текущая точка.
    p = -df(x0)  # Начальное направление — антиградиент.
    neval = 0  # Счётчик итераций (вычислений градиента).

    while True:
        gl = df(xmin)  # Градиент в текущей точке.
        # Линейный поиск по условиям Вульфа:
        alpha = wolfesearch(f, df, xmin, p, 3, tol, 0.1)
        xmin = xmin + alpha * p  # Обновление позиции.
        g = df(xmin)  # Новый градиент.

        #Beta = np.dot((g - gl).T, g) / np.dot(gl.T, gl)  # Полак-Рибьер
        Beta = np.dot(g.T, g) / np.dot(gl.T, gl)  # Флетчер-Ривз (используется).
        #Beta = -np.dot(g.T, (g - gl)) / np.dot(p.T, (g - gl))  # Хестенес-Штифель.
        # Beta = -np.dot(g.T, g) / np.dot(p.T, (g - gl))  # Дай-Юань.

        p = -g + Beta * p  # Обновление сопряжённого направления.

        coordinates.append(xmin)  # Сохранение новой точки.

        neval += 1

        # Условие остановки: малый градиент или превышение лимита итераций.
        if norm(g) < tol or neval >= 1000:
            break

    fmin = f(xmin)  # Значение функции в найденной точке.

    answer_ = [xmin, fmin, neval, coordinates]
    return answer_

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

def prDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0].flatten()  # Преобразует вектор-столбец в одномерный массив.
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i].flatten()
        x1 = coords[i + 1].flatten()
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)

    ax.text(x1[0] - 0.35, x1[1] - 0.2, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)

def draw(coords, nsteps, f, flag):
    fig, ax = plt.subplots()
    fig.suptitle('Polar Ribiere method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    prDraw(ax, coords, nsteps)
    contourPlot(ax, f)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def main():
    print("Himmelblau function:")
    x0 = np.array([[1.0], [0.0]])  # Начальная точка как вектор-столбец (2x1).
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fH, dfH, x0, tol)  # Минимизация Химмельблау.
    print(xmin, f, neval)
    draw(coords,  len(coords), fH, "h")

    print("Rosenbrock function:")
    x0 = np.array([[-1], [-2]])  # Начальная точка для Розенброка.
    tol = 1e-9
    [xmin, f, neval, coords] = prsearch(fR, dfR, x0, tol)  # Минимизация Розенброка.
    print(xmin, f, neval)
    draw(coords,  len(coords), fR, "r")

if __name__ == '__main__':
    main()