import numpy as np                     # Импортирует библиотеку NumPy для численных вычислений.
from numpy.linalg import norm          # Импортирует функцию norm для вычисления евклидовой нормы вектора.
import matplotlib                      # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')                  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI.
import matplotlib.pyplot as plt        # Импортирует pyplot из Matplotlib под псевдонимом plt.
import random                          # Импортирует модуль random (не используется в основном коде, но оставлен).

def goldensectionsearch(f, interval, tol):  # Определение функции поиска методом золотого сечения
    a, b = interval                        # Извлекает левую (a) и правую (b) границы интервала.
    gr = (1 + np.sqrt(5)) / 2              # Вычисляет золотое отношение φ ≈ 1.618.
    resphi = 2 - gr                        # Вычисляет 1/φ² ≈ 0.382 — коэффициент сужения.
    h = abs(b - a)                         # Вычисляет текущую длину интервала.
    if h <= tol:                           # Если интервал уже меньше допуска:
        xm = (a + b) / 2                   # Находит середину интервала.
        return [xm, f(xm), 0]              # Возвращает точку, значение функции и 0 вычислений (т.к. f(xm) — одно, но логика упрощена).
    c = a + resphi * h                     # Первая внутренняя точка (ближе к a).
    d = b - resphi * h                     # Вторая внутренняя точка (ближе к b).
    yc = f(c)                              # Значение функции в точке c.
    yd = f(d)                              # Значение функции в точке d.
    neval = 2                              # Счётчик вычислений функции (yc и yd).

    while h > tol:                         # Пока длина интервала больше допуска:
        if yc < yd:                        # Если f(c) < f(d), минимум слева → сужаем справа.
            b = d                          # Новая правая граница — d.
            d = c                          # Переносим c → d.
            yd = yc                        # Значение f(c) → f(d).
            h = abs(b - a)                 # Обновляем длину интервала.
            c = a + resphi * h             # Новая точка c.
            yc = f(c)                      # Вычисляем f(c).
        else:                              # Иначе минимум справа → сужаем слева.
            a = c                          # Новая левая граница — c.
            c = d                          # Переносим d → c.
            yc = yd                        # Значение f(d) → f(c).
            h = abs(b - a)                 # Обновляем длину интервала.
            d = b - resphi * h             # Новая точка d.
            yd = f(d)                      # Вычисляем f(d).
        neval += 1                         # Увеличиваем счётчик за одно новое вычисление.

    xm = (a + b) / 2                       # Финальное приближение — середина последнего интервала.
    fmin = f(xm)                           # Значение функции в xmin.
    neval += 1                             # Увеличиваем счётчик за последнее вычисление f(xm).
    return [xm, fmin, neval]               # Возвращает [минимум, значение, число вычислений].

def fR(X):                                 # Определение функции Розенброка
    x = X[0]                               # Извлекает x-координату.
    y = X[1]                               # Извлекает y-координату.
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # Вычисляет значение функции Розенброка.
    return v                               # Возвращает скалярное значение.

def dfR(X):                                # Определение градиента функции Розенброка
    x = X[0]                               # Извлекает x.
    y = X[1]                               # Извлекает y.
    v = np.copy(X)                         # Создаёт копию входного вектора.
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)  # ∂f/∂x.
    v[1] = 200 * (y - x ** 2)              # ∂f/∂y.
    return v                               # Возвращает градиент как вектор.

def bbsearch(f, df, x0, tol):
    # BBSEARCH searches for minimum using stabilized BB1 method
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]

    D = 0.1                                # Параметр стабилизации (ограничивает максимальный шаг).

    g = df(x0)                             # Градиент в начальной точке.
    func = lambda a: f(x0 - a * g)         # Одномерная функция для линейного поиска в направлении антиградиента.

    # Первый шаг: находим оптимальный α методом золотого сечения на [0, 1].
    alpha = goldensectionsearch(func, [0, 1], tol)[0]

    neval = 0                              # Счётчик итераций (не вычислений функции!).
    coords = [x0]                          # Траектория поиска.
    DeltaX = np.inf                        # Инициализация изменения позиции.

    while (neval < 1000) and (norm(DeltaX) > tol):  # Пока не достигнута точность и не превышено число итераций:
        xk = x0 - alpha * g                # Делает шаг в направлении антиградиента.
        g_new = df(xk)                     # Градиент в новой точке.
        DeltaX = xk - x0                   # Вектор изменения позиции.
        DeltaG = g_new - g                 # Вектор изменения градиента.

        # Формула BB1: α = (Δxᵀ·Δx) / (Δxᵀ·Δg)
        #alpha = np.dot(DeltaX.T, DeltaX) / np.dot(DeltaX.T, DeltaG)
        alpha = np.dot(DeltaX.T, DeltaG) / np.dot(DeltaG.T, DeltaG)  # 2

        # Стабилизирующий шаг: α_stab = D / ||g_new||
        alpha_stab = D / norm(g_new)       # ← Опечатка: "aplha" вместо "alpha"

        # Выбираем минимальный шаг для стабильности:
        alpha = min(alpha, alpha_stab)     # ← Использует опечатку

        x0 = xk                            # Обновляем текущую точку.
        g = g_new                          # Обновляем градиент.
        coords.append(xk)                  # Сохраняем точку в траекторию.

        neval += 1                         # Увеличиваем счётчик итераций.

    else:
        xmin = xk                          # Финальная точка.

    fmin = f(xmin)                         # Значение функции в минимуме.

    answer_ = [xmin, fmin, neval, coords]  # Формирует результат.
    return answer_                         # Возвращает результат.

def contourPlot(ax, f):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)           # Массив x от -4 до 4 с шагом 0.1.
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)           # Массив y от -4 до 4 с шагом 0.1.
    n = len(y1)

    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)         # Создаёт прямоугольную сетку координат.

    # массивы для графиков функции
    F = np.zeros((n, m))

    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)                 # Значение функции в точке сетки.

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)  # Рисует контурные линии.
    ax.set_xlabel('x')
    ax.set_ylabel('y')

#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def bbdDraw(ax, coords, nsteps):
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.1, x0[1] + 0.1, str(0), fontsize=fSize)  # Надпись "0" у старта.
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)  # Линия между шагами.

    ax.text(x1[0] + 0.2, x1[1], str(nsteps), fontsize=fSize)  # Надпись у финала.
    plt.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)  # Красная точка — минимум.

def draw(coords, nsteps, f):
    fig, ax = plt.subplots()
    fig.suptitle('Barzilai-Borwein 2 method each step visualisation & Countour plot')
    plt.xlim(-1, 3)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по осям.
    bbdDraw(ax, coords, nsteps)          # Рисует траекторию метода BB.
    contourPlot(ax, f)                   # Накладывает контуры функции.
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def main():
    print("Rosenbrock function:")
    x0 = np.array([2,-1])                # Начальная точка для функции Розенброка.
    tol = 1e-5                           # Очень высокая точность.
    [xmin, f, neval, coords] = bbsearch(fR, dfR, x0, tol)  # Запуск метода BB1.
    print(xmin, f, neval)
    draw(coords, len(coords), fR)        # Визуализация.

if __name__ == '__main__':
    main()