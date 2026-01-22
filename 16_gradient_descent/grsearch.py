from numpy.linalg import norm  # Импортирует функцию norm из numpy.linalg для вычисления евклидовой нормы вектора.
import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI (для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def f(X):
    x = X[0]  # Извлекает первую координату (x) из входного вектора X.
    y = X[1]  # Извлекает вторую координату (y) из входного вектора X.
    # Версия питона в codeboard не поддерживает метод библиотеки numpy float_power
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Вычисляет значение функции Химмельблау: f(x,y) = (x²+y−11)² + (x+y²−7)².
    return v  # Возвращает скалярное значение функции.

# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def df(X):
    x = X[0]  # Извлекает x.
    y = X[1]  # Извлекает y.
    v = np.copy(X)  # Создаёт копию вектора X для хранения градиента.
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)  # Частная производная по x: ∂f/∂x.
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)  # Частная производная по y: ∂f/∂y.
    return v  # Возвращает градиент как вектор [df/dx, df/dy].

def grsearch(x0, tol):
    # GRSEARCH searches for minimum using gradient descent method
    # 	answer_ = grsearch(x0,tol)
    #   INPUT ARGUMENTS
    #	x0 - starting point
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    #   answer_ = [xmin, fmin, neval, coords]
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of x values found during optimization

    al = 0.01  # Фиксированный шаг (learning rate) градиентного спуска.
    kmax = 1000  # Максимальное число итераций (защита от бесконечного цикла).

    x = x0  # Текущая точка начинается с начального приближения x0.
    coords = []  # Инициализирует список для хранения траектории поиска.
    coords.append(x)  # Добавляет начальную точку в траекторию.

    neval = 0  # Счётчик вычислений градиента (функции df).
    k = 0  # Счётчик итераций.
    deltaX = np.inf  # Инициализирует изменение позиции как бесконечность (для входа в цикл).

    while (norm(deltaX) >= tol) and (k < kmax):  # Пока изменение больше допуска И не превышено макс. число итераций:
        grad = df(x)  # Вычисляет градиент в текущей точке.
        neval += 1  # Увеличивает счётчик вычислений градиента.
        x_new = x - al * grad  # Делает шаг против градиента (спуск).
        deltaX = x_new - x  # Вычисляет вектор изменения позиции.
        x = x_new  # Обновляет текущую точку.
        coords.append(x)  # Сохраняет новую точку в траекторию.
        k += 1  # Увеличивает счётчик итераций.

    xmin = x  # Финальная точка — приближение к минимуму.
    fmin = f(x)  # Вычисляет значение функции в найденной точке.

    answer_ = [xmin, fmin, neval, coords]  # Формирует результат.
    return answer_  # Возвращает результат.

def contourPlot(ax):
    # Подготовка к рисованию, настраиваем оси x и y
    x1 = np.arange(-4, 4.1, 0.1)  # Создаёт массив значений x от -4 до 4 с шагом 0.1.
    m = len(x1)  # Количество точек по x.
    y1 = np.arange(-4, 4.1, 0.1)  # Создаёт массив значений y от -4 до 4 с шагом 0.1.
    n = len(y1)  # Количество точек по y.

    # делаем сетку
    [xx, yy] = np.meshgrid(x1, y1)  # Создаёт прямоугольную сетку координат (xx, yy).

    # массивы для графиков функции и ее производных по x и y
    F = np.zeros((n, m))  # Инициализирует матрицу значений функции.

    # вычисляем рельеф поверхности
    for i in range(n):  # Цикл по строкам (y).
        for j in range(m):  # Цикл по столбцам (x).
            X = [xx[i, j], yy[i, j]]  # Берёт координаты точки (x, y).
            F[i, j] = f(X)  # Вычисляет значение функции в этой точке.

    nlevels = 20  # Количество уровней контурных линий.
    ax.contour(xx, yy, F, nlevels, linewidths=1)  # Рисует контурные линии функции.
    ax.set_xlabel('x')  # Подписывает ось X.
    ax.set_ylabel('y')  # Подписывает ось Y.

#   - если не задавать цвет, то на итоговом графике видны шаги и маркер выглядит тогда лишним
# из минуса - нет возможности приближать график
def gradientDraw(ax, coords, nsteps):
    fSize = 11  # Размер шрифта для номеров итераций.
    x0 = coords[0]  # Начальная точка.
    ax.text(x0[0] + 0.1, x0[1] + 0.1, str(0), fontsize=fSize)  # Надпись "0" у начальной точки.
    for i in range(nsteps - 1):  # Цикл по всем шагам (соединяет точки линиями).
        x0 = coords[i]  # Текущая точка.
        x1 = coords[i + 1]  # Следующая точка.
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)  # Рисует отрезок между точками с квадратными маркерами.

    ax.text(x1[0] - 0.1, x1[1] - 0.5, str(nsteps), fontsize=fSize)  # Надпись с номером последней итерации.
    plt.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)  # Отмечает финальную точку красным кружком.

def draw(coords, nsteps):
    fig, ax = plt.subplots()  # Создаёт новую фигуру и оси.
    fig.suptitle('Gradient method each step visualisation & Countour plot')  # Заголовок фигуры.
    plt.xlim(-4, 4)  # Устанавливает пределы по оси X.
    plt.ylim(-4, 4)  # Устанавливает пределы по оси Y.
    plt.gca().set_aspect('equal', adjustable='box')  # Делает масштаб по осям одинаковым (квадратные ячейки).
    gradientDraw(ax, coords, nsteps)  # Рисует траекторию градиентного спуска.
    contourPlot(ax)  # Накладывает контурный график функции.
    name = "plot.png"  # Имя файла для сохранения.
    fig.savefig(name)  # Сохраняет график в файл.
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"  # Формирует HTML-тег для отображения.
    print(ad)  # Выводит HTML-тег.

def main():
    x0 = np.array([0, 1])  # Задаёт начальную точку поиска.
    tol = 1e-3  # Задаёт точность остановки (по норме изменения позиции).
    [xmin, f, neval, coords] = grsearch(x0, tol)  # Вызывает метод градиентного спуска.
    print(xmin, f, neval)  # Выводит найденный минимум, значение функции и число вычислений градиента.
    draw(coords, len(coords))  # Вызывает функцию визуализации траектории.

if __name__ == '__main__':
    main()  # Запускает основную функцию, если скрипт запущен напрямую.