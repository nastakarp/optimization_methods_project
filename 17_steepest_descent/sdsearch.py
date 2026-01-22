from numpy.linalg import norm  # Импортирует функцию norm для вычисления евклидовой нормы вектора.
import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI.
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt.

def fSphere(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    return np.sum(X ** 2)

def dfSphere(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return 2 * X

def fSumPowers(X):
    """
    Sum of Different Powers Function for d=2:
        f(x1, x2) = |x1|^2 + |x2|^3
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    return abs(X[0])**2 + abs(X[1])**3

def dfSumPowers(X):
    """
    Gradient of fSumPowers.
    grad = [ 2 * x1 * sign(x1), 3 * x2^2 * sign(x2) ]
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()

    x1, x2 = X[0], X[1]

    # Производная по x1: d/dx1 (|x1|^2) = 2 * |x1| * sign(x1) = 2 * x1
    # (потому что |x1|^2 = x1^2 → производная = 2*x1)
    g1 = 2 * x1

    # Производная по x2: d/dx2 (|x2|^3) = 3 * |x2|^2 * sign(x2) = 3 * x2 * |x2|
    g2 = 3 * x2 * abs(x2)

    return np.array([g1, g2])

def fZakharov(X):
    """
    Zakharov function for d=2:
        f(x1, x2) = x1^2 + x2^2 + (0.5*x1 + x2)^2 + (0.5*x1 + x2)^4
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]
    s = 0.5 * x1 + x2  # линейная комбинация
    return x1**2 + x2**2 + s**2 + s**4

def dfZakharov(X):
    """
    Gradient of Zakharov function.
    grad = [df/dx1, df/dx2]
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]
    s = 0.5 * x1 + x2

    # df/dx1 = 2*x1 + 2*s*(0.5) + 4*s^3*(0.5)
    # df/dx2 = 2*x2 + 2*s*(1)   + 4*s^3*(1)
    g1 = 2 * x1 + 2 * s * 0.5 + 4 * s**3 * 0.5
    g2 = 2 * x2 + 2 * s * 1   + 4 * s**3 * 1

    return np.array([g1, g2])

def fMcCormick(X):
    """
    McCormick function:
        f(x1, x2) = sin(x1 + x2) + (x1 - x2)^2 - 1.5*x1 + 2.5*x2 + 1
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1

def dfMcCormick(X):
    """
    Gradient of McCormick function.
    grad = [df/dx1, df/dx2]
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]

    # df/dx1 = cos(x1+x2) + 2*(x1-x2)*1 - 1.5
    # df/dx2 = cos(x1+x2) + 2*(x1-x2)*(-1) + 2.5
    g1 = np.cos(x1 + x2) + 2 * (x1 - x2) - 1.5
    g2 = np.cos(x1 + x2) - 2 * (x1 - x2) + 2.5

    return np.array([g1, g2])

def fBranin(X):
    """
    Branin function with standard parameters.
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    Global minima at: (-π, 12.275), (π, 2.275), (9.42478, 2.475) → f ≈ 0.397887
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]

    # Параметры
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)

    # Вычисление
    term1 = x2 - b * x1**2 + c * x1 - r
    term2 = s * (1 - t) * np.cos(x1)
    return a * term1**2 + term2 + s

def dfBranin(X):
    """
    Gradient of Branin function.
    grad = [df/dx1, df/dx2]
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]

    # Параметры
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)

    term1 = x2 - b * x1**2 + c * x1 - r

    # df/dx1 = 2a * term1 * (-2b*x1 + c) - s*(1-t)*sin(x1)
    g1 = 2 * a * term1 * (-2 * b * x1 + c) - s * (1 - t) * np.sin(x1)

    # df/dx2 = 2a * term1 * 1
    g2 = 2 * a * term1

    return np.array([g1, g2])

def fMatyas(X):
    """
    Matyas function:
        f(x1, x2) = 0.26*(x1^2 + x2^2) - 0.48*x1*x2
    Global minimum at (0, 0), f(0,0) = 0.
    Convex, smooth, bowl-shaped.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def dfMatyas(X):
    """
    Gradient of Matyas function.
    grad = [df/dx1, df/dx2]
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if X.shape == (2, 1):
        X = X.flatten()
    x1, x2 = X[0], X[1]

    # df/dx1 = 0.26 * 2 * x1 - 0.48 * x2
    # df/dx2 = 0.26 * 2 * x2 - 0.48 * x1
    g1 = 0.52 * x1 - 0.48 * x2
    g2 = 0.52 * x2 - 0.48 * x1

    return np.array([g1, g2])

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

'''
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
'''

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
    x1 = np.arange(-4, 11.1, 0.1)  # Массив x от -4 до 4.
    m = len(x1)
    y1 = np.arange(-4, 11.1, 0.1)  # Массив y от -4 до 4.
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
    plt.xlim(-4, 11)
    plt.ylim(-4, 11)
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по осям.
    steepDraw(ax, coords, nsteps)  # Рисует траекторию.
    contourPlot(ax, f)  # Накладывает контуры функции.
    name = "plot" + flag + ".png"  # Имя файла: "ploth.png" или "plotr.png".
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def main():
    print("SumPowers function:")
    x0 = np.array([1.3, 2.0])
    tol = 1e-3
    [xmin, fmin, neval, coords] = sdsearch(fMatyas, dfMatyas, x0, tol)
    print(xmin, fmin, neval)
    draw(coords, len(coords), "m", fMatyas)


if __name__ == '__main__':
    main()  # Запускает основную функцию.