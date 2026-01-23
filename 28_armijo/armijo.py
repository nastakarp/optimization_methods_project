import math  # Импортирует модуль math для математических функций (хотя в коде не используется).
from numpy.linalg import norm  # Импортирует функцию norm для вычисления евклидовой нормы вектора.
import numpy as np  # Импортирует библиотеку NumPy для числовых операций.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' для сохранения графиков без GUI.
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt.

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def F(X):
    x = X[0]  # Извлекает координату x из входного вектора X.
    y = X[1]  # Извлекает координату y из входного вектора X.
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Вычисляет значение функции Химмельблау.
    return v  # Возвращает скалярное значение функции.

# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dF(X):
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

'''def fR(X):  # Закомментировано: функция Розенброка не используется в этом скрипте.
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v'''

# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

'''def dfR(X):  # Закомментировано: градиент Розенброка не используется.
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v'''

def armijo(f, df, x0, p0, a, c1, b):
    # Реализует линейный поиск по правилу Армихо.
    phi = lambda x: f(x0 + x * p0)  # Одномерная функция вдоль направления p0.
    dphi = lambda x: np.dot(p0.transpose(), df(x0 + x * p0))  # Её производная.

    phi0 = phi(0)  # Значение функции в начальной точке.
    dphi0 = dphi(0)  # Производная в начальной точке.

    k = 0
    # Пока условие Армихо не выполнено и не превышен лимит итераций:
    while phi(a) > phi0 + c1 * a * dphi0 and k < 1000:
        a = a * b  # Уменьшаем шаг (геометрическая прогрессия).
        k += 1

    return a  # Возвращает найденный шаг.

def prsearch(f, df, x0, tol):
    # PRSEARCH searches for minimum using Polak-Ribiere method
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]

    coordinates = [x0]  # Список для хранения траектории поиска.
    xmin = x0  # Текущая точка.
    p = -df(x0)  # Начальное направление — антиградиент.
    neval = 0  # Счётчик итераций.

    while True:
        gl = df(xmin)  # Градиент в текущей точке.
        alpha = armijo(f, df, xmin, p, 1, 0.1, 0.5)  # Используется правило Армихо.
        xmin = xmin + alpha * p  # Обновление позиции.
        g = df(xmin)  # Новый градиент.

        # Формула обновления коэффициента Beta для метода Полака-Рибьера:
        Beta = np.dot((g - gl).T, g) / np.dot(gl.T, gl)
        # Другие закомментированные формулы:
        # Beta = np.dot(g.T, g) / np.dot(gl.T, gl)  # Флетчер-Ривз.
        # Beta = -np.dot(g.T, (g - gl)) / np.dot(p.T, (g - gl))  # Хестенес-Штифель.
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
    # Рисует контурный график функции f.
    x1 = np.arange(-4, 4.1, 0.1)
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)
    n = len(y1)

    # Создаёт прямоугольную сетку координат:
    [xx, yy] = np.meshgrid(x1, y1)

    # Инициализирует матрицу для значений функции:
    F = np.zeros((n, m))

    # Вычисляет значение функции в каждой точке сетки:
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
    x0 = coords[0].flatten()
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
    # Основная функция программы.
    print("test function:")  # Выводит заголовок.
    x0 = np.array([[0.0], [10.0]])  # Задаёт начальную точку как вектор-столбец.
    tol = 1e-9  # Задаёт высокую точность остановки.
    [xmin, f, neval, coords] = prsearch(F, dF, x0, tol)  # Запускает метод Полака-Рибьера для функции Химмельблау.
    print(xmin, f, neval)  # Выводит результат.
    draw(coords, len(coords), f, "t")  # Создаёт и выводит график.

if __name__ == '__main__':
    main()  # Запускает основную функцию.