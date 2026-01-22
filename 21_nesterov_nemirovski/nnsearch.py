import numpy as np                     # Импортирует библиотеку NumPy для численных вычислений.
import sys                             # Импортирует модуль sys (не используется в основном коде).
from numpy.linalg import norm          # Импортирует функцию norm для вычисления евклидовой нормы вектора.
from pyparsing import alphas           # ОШИБКА: импорт из pyparsing конфликтует с локальной переменной `alphas`! (см. замечание ниже)
np.seterr(all='warn')                  # Настройка обработки ошибок NumPy: генерировать предупреждения вместо исключений.
import numpy as np                     # Повторный импорт (избыточен, но безвреден).
import matplotlib                      # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')                  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI.
import matplotlib.pyplot as plt        # Импортирует pyplot из Matplotlib под псевдонимом plt.
import random                          # Импортирует модуль random (не используется в основном коде).

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]                           # Извлекает x из вектора X.
    y = X[1]                           # Извлекает y из вектора X.
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Вычисляет значение функции Химмельблау.
    return v                           # Возвращает скалярное значение.

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
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)      # ∂f/∂x.
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)      # ∂f/∂y.
    return v                           # Возвращает градиент.

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
    v[1] = 200 * (y - x ** 2)            # ∂f/∂y.
    return v

def H(X, tol, df):
    n = X.size                         # Определяет размерность задачи (n=2).
    ddf = np.zeros((n, n))             # Инициализирует матрицу Гессе нулями.

    delta = 0.1 * tol                  # Шаг для численного дифференцирования.
    dfx0 = df(X)                       # Градиент в базовой точке X.
    df_values = []                     # Список для хранения градиентов в возмущённых точках.

    for i in range(n):                 # Цикл по переменным:
        new_x = X.copy()               # Копия X.
        new_x[i, 0] += delta           # Добавляет возмущение к i-й компоненте (X — столбец!).
        df_values.append(df(new_x))    # Сохраняет градиент в возмущённой точке.

    for i in range(n):                 # Цикл по строкам матрицы Гессе.
        for j in range(i, n):          # Цикл по столбцам (только верхний треугольник).
            # Аппроксимация частной производной ∂²f/∂x_i∂x_j через прямую разность:
            ddf[i, j] = (df_values[j][i, 0] - dfx0[i, 0]) / delta
            if i != j:                 # Матрица Гессе симметрична:
                ddf[j, i] = ddf[i, j]
    return ddf                         # Возвращает приближённую матрицу Гессе.

def nnsearch(f, df, x0, tol):
    # Реализует модифицированный метод Ньютона с адаптивным демпфированием.
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point (столбец!)
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords, alphas]

    kmax = 1000                        # Максимальное число итераций.
    deltaX = np.array([10 ** 6] * 2)   # Инициализация большого изменения для входа в цикл.

    k = 0                              # Счётчик итераций.
    neval = 0                          # Счётчик вычислений.

    coords = [x0.copy()]               # Траектория поиска (список столбцов).
    alphas = []                        # Список коэффициентов демпфирования (шагов).
    x = np.array(x0, dtype=float)      # Текущая точка (копия начальной).

    while (norm(deltaX) >= tol) and (k < kmax):
        Hm = H(x, tol, df)             # Вычисляет численную матрицу Гессе в точке x.
        neval += len(x0) + 1           # Приблизительный счётчик за вычисление Гессе.
        g = df(x)                      # Градиент в текущей точке.

        # Вычисляет "размер" шага Ньютона в метрике Гессе:
        delt = np.sqrt(g.T @ np.linalg.solve(Hm, g))
        if delt <= 0.25:               # Если шаг мал — используем полный шаг Ньютона.
            alpha = 1
        else:                          # Иначе — уменьшаем шаг.
            alpha = 1 / (1 + delt)

        neval += 1                     # Счётчик за вычисление градиента (уже был, но оставлен).

        deltaX = np.linalg.lstsq(-Hm, g, rcond=None)[0]  # Направление Ньютона.
        x += alpha * deltaX            # Делает демпфированный шаг.
        coords.append(x.copy())        # Сохраняет новую точку.
        alphas.append(alpha)           # Сохраняет коэффициент шага.
        k += 1

    xmin = x                           # Финальная точка.
    fmin = f(xmin)                     # Значение функции.
    neval += 1                         # Счётчик за последнее вычисление f.

    answer_ = [xmin, fmin, neval, coords, alphas]  # Формирует результат.
    return answer_

# Эта функция определена, но НЕ ИСПОЛЬЗУЕТСЯ (переопределена ниже). Можно удалить.
def newtonDraw(coords, nsteps, flag):
    fig = plt.figure()
    fig.suptitle('Newton method each step visualisation')
    fSize = 11
    x0 = coords[0]
    plt.text(x0[0] + 0.2, x0[1] - 0.1, str(0), fontsize=fSize)
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        plt.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)
    plt.text(x1[0] + 0.2, x1[1] - 0.1, str(nsteps), fontsize=fSize)
    plt.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)
    plt.ylim(top=x1[1] + 0.5)
    name = "plot" + flag + ".png"
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def contourPlot(ax, f):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)       # Массив x от -4 до 4.
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)       # Массив y от -4 до 4.
    n = len(y1)
    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)     # Создаёт прямоугольную сетку координат.
    # массивы для графиков функции
    F = np.zeros((n, m))

    # вычисляем рельеф поверхности
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]
            F[i, j] = f(X)             # Вычисляет значение функции в каждой точке сетки.

    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)  # Рисует контурные линии.
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# ПЕРЕОПРЕДЕЛЕННАЯ функция — используется в draw().
# Обратите внимание: ожидает, что coords[i] — это столбец (2x1)!
def newtonDraw(ax, coords, alphas, nsteps):
    fSize = 11
    x0 = coords[0]
    # Извлекает скалярные значения из столбца:
    ax.text(x0[0, 0] + 0.03, x0[1, 0] + 0.1, str(0), fontsize=fSize)

    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]

        # Цвет шага зависит от коэффициента демпфирования:
        color = 'blue' if alphas[i] == 1 else 'green'

        ax.plot(
            [x0[0, 0], x1[0, 0]],     # x-координаты
            [x0[1, 0], x1[1, 0]],     # y-координаты
            lw=1.2,
            marker='s',
            ms=0.2,
            color=color                # Синий — полный шаг, зелёный — уменьшенный.
        )

    ax.text(x1[0, 0] - 0.2, x1[1, 0] - 0.25, str(nsteps), fontsize=fSize)
    ax.scatter(x1[0, 0], x1[1, 0], marker='o', c='red', zorder=12)

def draw(coords, alphas, nsteps, flag, f):
    fig, ax = plt.subplots()
    fig.suptitle('Newton method each step visualisation & Countour plot')

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по осям.

    newtonDraw(ax, coords, alphas, nsteps)  # Рисует траекторию с цветовой индикацией шагов.
    contourPlot(ax, f)                      # Накладывает контурный график функции.

    name = "plot" + flag + ".png"
    fig.savefig(name)
    print(f'<img width="900px" src="/resources/{name}">')

def main():
    print("Rosenbrock function:")
    # ВАЖНО: x0 задан как столбец (2x1 матрица), а не вектор!
    x0 = np.array([[-2.0], [-2.0]])
    tol = 1e-9                         # Очень высокая точность для функции Розенброка.
    [xmin, f, neval, coords, alphas] = nnsearch(fR, dfR, x0, tol)  # Запуск модифицированного Ньютона.
    print(xmin, f, neval)
    draw(coords, alphas, len(coords), "r", fR)  # Визуализация с цветовой индикацией шагов.

if __name__ == '__main__':
    main()