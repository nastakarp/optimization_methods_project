from numpy.linalg import norm  # Импортирует функцию norm для вычисления евклидовой нормы вектора.
import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI.
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt.

def goldensectionsearch(f, interval, tol):
    a, b = interval  # Распаковывает интервал [a, b].
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение φ ≈ 1.618.
    resphi = 2 - phi  # Обратное значение: 1/φ² ≈ 0.382 — коэффициент сужения.
    neval = 0  # Счётчик вычислений функции f.

    x1 = a + resphi * (b - a)  # Первая внутренняя точка.
    x2 = b - resphi * (b - a)  # Вторая внутренняя точка (симметрична x1).
    f1 = f(x1)  # Значение функции в x1.
    f2 = f(x2)  # Значение функции в x2.
    neval += 2  # Увеличивает счётчик за два вычисления.

    while abs(b - a) > tol:  # Пока длина интервала больше допуска:
        if f1 < f2:  # Если f(x1) < f(x2), минимум слева → сужаем справа.
            b = x2  # Новая правая граница — x2.
            x2 = x1  # Переносим x1 → x2.
            f2 = f1  # Значение f1 → f2.
            x1 = a + resphi * (b - a)  # Новая точка x1.
            f1 = f(x1)  # Вычисляем f(x1).
        else:  # Иначе минимум справа → сужаем слева.
            a = x1  # Новая левая граница — x1.
            x1 = x2  # Переносим x2 → x1.
            f1 = f2  # Значение f2 → f1.
            x2 = b - resphi * (b - a)  # Новая точка x2.
            f2 = f(x2)  # Вычисляем f(x2).
        neval += 1  # Увеличивает счётчик за одно новое вычисление.

    xmin = (a + b) / 2  # Финальное приближение — середина последнего интервала.
    fmin = f(xmin)  # Значение функции в xmin.
    answer_ = [xmin, fmin, neval]  # Формирует результат.
    return answer_  # Возвращает [минимум, значение, число вычислений].

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]  # Извлекает x.
    y = X[1]  # Извлекает y.
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Функция Химмельблау.
    return v  # Возвращает скаляр.

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
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)  # ∂f/∂x.
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)  # ∂f/∂y.
    return v  # Возвращает градиент.

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

def sdsearch(f, df, x0, tol):
    # SDSEARCH searches for minimum using steepest descent method
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]

    kmax = 1000  # Максимальное число итераций.
    x = x0  # Текущая точка.
    coords = []  # Список для хранения траектории.
    coords.append(x)  # Добавляет начальную точку.
    neval = 0  # Счётчик вычислений функции f (включая линейный поиск).
    k = 0  # Счётчик итераций.
    deltaX = np.inf  # Инициализация изменения позиции.

    while (norm(deltaX) >= tol) and (k < kmax):  # Пока изменение велико и не превышено kmax:
        grad = df(x)  # Вычисляет градиент в текущей точке.
        neval += 1  # Считаем как одно вычисление (хотя df может вызывать f, но здесь считаем отдельно).

        # Одномерная функция для линейного поиска: f(x - α·grad)
        f1dim = lambda alpha: f(x - alpha * grad)

        interval = [0.0, 1.0]  # Интервал поиска шага α.
        alpha_opt, _, neval_1d = goldensectionsearch(f1dim, interval, tol)  # Находит оптимальный α.
        neval += neval_1d  # Добавляет число вычислений из линейного поиска.

        x_new = x - alpha_opt * grad  # Делает шаг с оптимальным α.
        deltaX = x_new - x  # Вектор изменения.

        coords.append(x_new)  # Сохраняет новую точку.
        x = x_new  # Обновляет текущую точку.
        k += 1  # Увеличивает счётчик итераций.

    xmin = x  # Финальная точка.
    fmin = f(xmin)  # Значение функции в минимуме.

    answer_ = [xmin, fmin, neval, coords]  # Формирует результат.
    return answer_  # Возвращает результат.

def contourPlot(ax, f):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)  # Массив x от -4 до 4.
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)  # Массив y от -4 до 4.
    n = len(y1)
    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)  # Создаёт сетку координат.
    # массивы для графиков функции
    F = np.zeros((n, m))

    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)  # Вычисляет значение функции в каждой точке сетки.

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)  # Рисует контурные линии.
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def steepDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)  # Надпись "0" у старта.
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=0.2)  # Линия между шагами.

    ax.text(x1[0] + 0.1, x1[1] - 0.2, str(nsteps), fontsize=fSize)  # Надпись у финала.
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)  # Красная точка — минимум.

def draw(coords, nsteps, flag, f):
    fig, ax = plt.subplots()
    fig.suptitle('Steepest descent method each step visualisation & Countour plot')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по осям.
    steepDraw(ax, coords, nsteps)  # Рисует траекторию.
    contourPlot(ax, f)  # Накладывает контуры функции.
    name = "plot" + flag + ".png"  # Имя файла: "ploth.png" или "plotr.png".
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def main():
    print("Himmelblau function:")
    x0 = np.array([1.3, 2.0])  # Начальная точка для Химмельблау.
    tol = 1e-3
    [xmin, fmin, neval, coords] = sdsearch(fH, dfH, x0, tol)
    print(xmin, fmin, neval)
    draw(coords, len(coords), "h", fH)  # Визуализация для Химмельблау.

    print("Rosenbrock function:")
    x0 = np.array([1.0, -2.0])  # Начальная точка для Розенброка.
    tol = 1e-7  # Более высокая точность (функция "плоская").
    [xmin, fmin, neval, coords] = sdsearch(fR, dfR, x0, tol)
    print(xmin, fmin, neval)
    draw(coords, len(coords), "r", fR)  # Визуализация для Розенброка.

if __name__ == '__main__':
    main()  # Запускает основную функцию.