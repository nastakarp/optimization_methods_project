import numpy as np                     # Импорт библиотеки NumPy для численных вычислений.
from numpy.linalg import norm          # Импорт функции norm для вычисления евклидовой нормы вектора.
np.seterr(all='warn')                  # Настройка обработки ошибок NumPy: генерировать предупреждения вместо исключений при переполнении/делении на ноль.
import matplotlib                      # Импорт основного модуля Matplotlib для визуализации.
matplotlib.use('Agg')                  # Установка backend 'Agg' — позволяет сохранять графики без GUI (для серверов/headless-режима).
import matplotlib.pyplot as plt        # Импорт pyplot из Matplotlib под псевдонимом plt.
import numpy as np                     # Повторный импорт (избыточен, но безвреден).

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]                           # Извлекает первую координату (x) из входного вектора X.
    y = X[1]                           # Извлекает вторую координату (y) из входного вектора X.
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2  # Вычисляет значение функции Химмельблау: f(x,y) = (x²+y−11)² + (x+y²−7)².
    return v                           # Возвращает скалярное значение функции.

# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dfH(X):
    x = X[0]                           # Извлекает x.
    y = X[1]                           # Извлекает y.
    v = np.copy(X)                     # Создаёт копию вектора X для хранения градиента.
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)      # Частная производная по x: ∂f/∂x.
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)      # Частная производная по y: ∂f/∂y.
    return v                           # Возвращает градиент как вектор [df/dx, df/dy].

# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fR(X):
    x = X[0]                           # Извлекает x.
    y = X[1]                           # Извлекает y.
    v = (1 - x)**2 + 100*(y - x**2)**2 # Вычисляет значение функции Розенброка ("банановая долина").
    return v                           # Возвращает скалярное значение функции.

# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def dfR(X):
    x = X[0]                           # Извлекает x.
    y = X[1]                           # Извлекает y.
    v = np.copy(X)                     # Создаёт копию вектора X для хранения градиента.
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)  # Частная производная по x: ∂f/∂x.
    v[1] = 200 * (y - x**2)            # Частная производная по y: ∂f/∂y.
    return v                           # Возвращает градиент.

def H(X, tol, df):
    n = len(X)                         # Определяет размерность задачи (n=2 для двумерных функций).
    Hx = np.zeros((n, n))              # Инициализирует матрицу Гессе (n×n) нулями.
    h = tol                            # Использует допуск tol как шаг для численного дифференцирования.

    for j in range(n):                 # Цикл по столбцам матрицы Гессе (∂²f/∂x_i∂x_j по j).
        x_plus = np.copy(X)            # Копия X для положительного возмущения.
        x_minus = np.copy(X)           # Копия X для отрицательного возмущения.

        x_plus[j] += h                 # Добавляет шаг h к j-й компоненте.
        x_minus[j] -= h                # Вычитает шаг h из j-й компоненты.

        df_plus = df(x_plus)           # Градиент в точке с положительным возмущением.
        df_minus = df(x_minus)         # Градиент в точке с отрицательным возмущением.

        for i in range(n):             # Цикл по строкам матрицы Гессе (∂²f/∂x_i∂x_j по i).
            # Центральная разность для аппроксимации второй частной производной:
            Hx[i, j] = (df_plus[i] - df_minus[i]) / (2 * h)

    return Hx                          # Возвращает приближённую матрицу Гессе.

def nsearch(f, df, x0, tol):
    # NSEARCH searches for minimum using Newton method
    #   INPUT ARGUMENTS
    #   f  - objective function
    #   df - gradient
    #   x0 - start point
    #   tol - tolerance
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]

    kmax = 1000                        # Максимальное число итераций (защита от зацикливания).
    deltaX = np.array([10 ** 6] * 2)   # Инициализация изменения позиции большим значением (для входа в цикл).

    k = 0                              # Счётчик итераций.
    neval = 0                          # Счётчик вычислений функции/градиента.

    coords = [x0.copy()]               # Инициализирует список траектории начальной точкой.
    x = np.array(x0, dtype=float)      # Преобразует начальную точку в массив с плавающей точкой.

    while (norm(deltaX) >= tol) and (k < kmax):  # Пока изменение велико и не превышено kmax:
        Hm = H(x, tol, df)             # Вычисляет численную матрицу Гессе в точке x.
        neval += len(x0) + 1           # Увеличивает счётчик: len(x0) за df_plus/df_minus + 1 за текущий df.
        g = df(x)                      # Вычисляет градиент в текущей точке.
        neval += 1                     # Увеличивает счётчик за вычисление градиента.

        # Решает систему H·Δx = -g методом наименьших квадратов (устойчиво даже если H вырождена):
        deltaX = np.linalg.lstsq(-Hm, g, rcond=None)[0]
        x += deltaX                    # Обновляет текущую точку.
        coords.append(x.copy())        # Сохраняет копию точки в траекторию.
        k += 1                         # Увеличивает счётчик итераций.

    xmin = x                           # Финальная точка — приближение к минимуму.
    fmin = f(xmin)                     # Вычисляет значение функции в найденной точке.
    neval += 1                         # Увеличивает счётчик за последнее вычисление f.

    answer_ = [xmin, fmin, neval, coords]  # Формирует результат.
    return answer_                     # Возвращает результат.

def newtonDraw(coords, nsteps, flag):
    # Эта функция определена, но НЕ ИСПОЛЬЗУЕТСЯ (переопределена ниже). Можно удалить.
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
    x1 = np.arange(-4, 4.1, 0.1)       # Массив x от -4 до 4 с шагом 0.1.
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)       # Массив y от -4 до 4 с шагом 0.1.
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

#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def newtonDraw(ax, coords, nsteps):
    # ПЕРЕОПРЕДЕЛЕННАЯ функция — используется в draw().
    fSize = 11
    x0 = coords[0]
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)  # Надпись "0" у начальной точки.
    for i in range(nsteps - 1):
        x0 = coords[i]
        x1 = coords[i + 1]
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=0.2)  # Линия между шагами.
    ax.text(x1[0] - 0.2, x1[1] - 0.25, str(nsteps), fontsize=fSize)  # Надпись у финальной точки.
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)  # Красная точка — минимум.

def draw(coords, nsteps, flag, f):
    fig, ax = plt.subplots()           # Создаёт новую фигуру и оси.
    fig.suptitle('Newton method each step visualisation & Countour plot')
    plt.xlim(-4, 4)                    # Устанавливает пределы по X.
    plt.ylim(-4, 4)                    # Устанавливает пределы по Y.
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб по осям.
    newtonDraw(ax, coords, nsteps)     # Рисует траекторию метода Ньютона.
    contourPlot(ax, f)                 # Накладывает контурный график функции.
    name = "plot" + flag + ".png"      # Имя файла: "ploth.png" или "plotr.png".
    fig.savefig(name)
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"
    print(ad)

def main():
    print("Himmelblau function:")
    x0 = np.array([-2.0, -2.0])        # Начальная точка для функции Химмельблау.
    tol = 1e-3                         # Точность остановки.
    [xmin, f, neval, coords] = nsearch(fH, dfH, x0, tol)  # Запуск метода Ньютона.
    print(xmin, f, neval)              # Вывод результата.
    draw(coords, len(coords), "h", fH) # Визуализация для Химмельблау.

    print("Rosenbrock function:")
    x0 = np.array([-1.0,-1.0])         # Начальная точка для функции Розенброка.
    tol = 1e-9                         # Очень высокая точность (функция "плоская" около минимума).
    [xmin, f, neval, coords] = nsearch(fR, dfR, x0, tol)  # Запуск метода Ньютона.
    print(xmin, f, neval)              # Вывод результата.
    draw(coords, len(coords), "r", fR) # Визуализация для Розенброка.

if __name__ == '__main__':
    main()                             # Запускает основную функцию, если скрипт запущен напрямую.