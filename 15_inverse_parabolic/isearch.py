import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI (для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.
import random  # Импортирует модуль random для генерации случайных цветов при визуализации.

def f(x): return x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20  # Определяет целевую функцию f(x) — осциллирующая квадратичная функция с множеством локальных минимумов.

def df(x): return 2 * x + 3 * np.pi * np.sin(0.3 * np.pi * x)  # Определяет аналитическую производную f'(x).


def isearch(interval, tol):
    a, b = interval  # Распаковывает входной интервал [a, b].
    x0 = (a + b) / 2  # Берёт начальную точку как середину интервала.
    coords = [[x0, a, b]]  # Инициализирует список для хранения истории: [текущая точка, левая граница, правая граница].
    neval = 0  # Счётчик вычислений производной.

    xl = x0  # Текущая точка поиска.

    while True:  # Бесконечный цикл — выход по условию внутри.
        dfa = df(a)  # Вычисляет производную в левой границе.
        dfb = df(b)  # Вычисляет производную в правой границе.
        dfx = df(xl)  # Вычисляет производную в текущей точке.

        # Формула обратной интерполяции Лагранжа: находит x, при котором f'(x) = 0,
        # используя три точки (a, dfa), (b, dfb), (xl, dfx).
        xmin = ((dfb * dfx) / ((dfa - dfb) * (dfa - dfx))) * a + \
               ((dfa * dfx) / ((dfb - dfa) * (dfb - dfx))) * b + \
               ((dfa * dfb) / ((dfx - dfa) * (dfx - dfb))) * xl

        # Обновляет интервал, исходя из положения нового приближения:
        if xmin < xl:
            b = xl  # Новое приближение слева → сужаем справа.
        else:
            a = xl  # Новое приближение справа → сужаем слева.

        neval += 3  # Увеличивает счётчик за три вычисления df (a, b, xl).
        coords.append([xmin, a, b])  # Сохраняет новое состояние.

        if abs(xmin - xl) < tol:  # Если изменение меньше допуска — завершаем.
            break
        else:
            xl = xmin  # Иначе обновляем текущую точку.

    fmin = f(xmin)  # Вычисляет значение функции в найденной точке.

    answer_ = [xmin, fmin, neval, coords]  # Формирует результат.
    return answer_  # Возвращает результат.

# ---------- Базовые графики ----------
def drawdf(a, b, h, ax1):
    x = np.arange(a, b, h)  # Создаёт массив x от a до b с шагом h.
    y = [df(i) for i in x]  # Вычисляет значения производной f'(x).
    ax1.plot(x, y, lw=1, c='black')  # Рисует график f'(x) чёрной линией.
    ax1.axhline(0, color='black', lw=1)  # Горизонтальная линия y=0.
    ax1.set_xlabel('x')  # Подписывает ось X.
    ax1.set_ylabel("f'(x)")  # Подписывает ось Y как производную.
    ax1.set_xlim([a - 0.2, b + 0.2])  # Устанавливает пределы по X с небольшим отступом.

def drawf(a, b, h, ax2):
    x = np.arange(a, b, h)  # Массив x.
    y = [f(i) for i in x]  # Значения функции f(x).
    ax2.plot(x, y, lw=1, c='black')  # График f(x).
    ax2.set_xlabel('x')  # Подпись X.
    ax2.set_ylabel("f(x)")  # Подпись Y.
    ax2.set_xlim([a - 0.2, b + 0.2])  # Пределы по X.

# ---------- Обратный полином Лагранжа ----------
def inverse_lagrange(y, x0, x1, x2, y0, y1, y2):
    # Реализует формулу обратной интерполяции Лагранжа:
    # Находит x, соответствующее заданному y, зная три точки (x0,y0), (x1,y1), (x2,y2).
    return (
        x0 * (y - y1) * (y - y2) / ((y0 - y1) * (y0 - y2)) +
        x1 * (y - y0) * (y - y2) / ((y1 - y0) * (y1 - y2)) +
        x2 * (y - y0) * (y - y1) / ((y2 - y0) * (y2 - y1))
    )

# ---------- Итерационные точки и ПРАВИЛЬНАЯ парабола ----------
def drawpointsDF(coord, ax1, i):
    xk, a, b = coord  # Распаковывает текущую точку и границы интервала.
    color = (random.random(), random.random(), random.random())  # Случайный цвет.

    # значения производной в трёх точках:
    ya = df(a)
    yb = df(b)
    yk = df(xk)

    # Отмечает три точки на графике f'(x):
    ax1.scatter([a, xk, b], [ya, yk, yb], c=color, zorder=3)
    ax1.plot(xk, yk, marker='*', ms=10, c=color)  # Текущая точка — звезда.
    ax1.text(xk, yk + 2, str(i + 1))  # Номер итерации над точкой.

    # Вертикальная линия от точки до оси X:
    ax1.plot([xk, xk], [yk, 0], c=color, lw=1)

    # ---- обратная парабола x = P(df) ----
    y_min = min(ya, yb, yk)  # Минимальное значение производной среди трёх точек.
    y_max = max(ya, yb, yk)  # Максимальное значение.
    y_vals = np.linspace(y_min, y_max, 200)  # Плотная сетка по y.

    # Вычисляет x для каждого y по формуле обратного Лагранжа:
    x_vals = inverse_lagrange(
        y_vals,
        a, b, xk,  # x-координаты точек
        ya, yb, yk  # y-координаты (значения производной)
    )

    # Рисует обратную параболу (x как функция от y):
    ax1.plot(x_vals, y_vals, lw=2, ls='--', c=color)

    # Точка пересечения с осью f'(x)=0 (ключевая — следующее приближение):
    xmin = inverse_lagrange(
        0,  # y = 0
        a, b, xk,
        ya, yb, yk
    )

    # Отмечает эту точку на оси X:
    ax1.plot(xmin, 0, marker='o', ms=7, c=color)

def drawpointsF(coord, ax2, i):
    xk, a, b = coord
    color = (random.random(), random.random(), random.random())

    # Отмечает текущую точку на графике f(x):
    ax2.plot(xk, f(xk), marker='*', ms=10, c=color)
    ax2.text(xk, f(xk) + 1.2, str(i + 1))  # Номер итерации.
    ax2.scatter([a, b], [f(a), f(b)], c=color)  # Границы интервала.

# ---------- Основная визуализация ----------
def opisearchslides(coords, interval):
    fig = plt.figure(figsize=(10, 8))  # Создаёт фигуру размером 10×8 дюймов.
    ax1 = fig.add_subplot(2, 1, 1)  # Верхний график — для f'(x).
    ax2 = fig.add_subplot(2, 1, 2)  # Нижний график — для f(x).

    fig.suptitle('Inverse Parabolic Interpolation (Lagrange form)')  # Общий заголовок.

    a, b = interval  # Исходный интервал.
    h = (b - a) / 300  # Шаг дискретизации (более мелкий, чем раньше).

    drawdf(a, b, h, ax1)  # Рисует базовый график производной.
    drawf(a, b, h, ax2)  # Рисует базовый график функции.

    fig.savefig("plot.png")  # Сохраняет базовый график.
    print('<img width="900px" src="/resources/plot.png">')  # HTML-тег для отображения.

    nfigs = min(10, len(coords))  # Ограничивает число визуализируемых итераций до 10.
    filenames = []  # Список имён файлов.

    for i in range(nfigs):  # Цикл по итерациям.
        drawpointsDF(coords[i], ax1, i)  # Добавляет точки и обратную параболу на график f'(x).
        drawpointsF(coords[i], ax2, i)  # Добавляет точки на график f(x).

        ax1.set_title(f"Iteration {i + 1}")  # Заголовок верхнего графика.
        name = f"iter{i + 1}.png"
        filenames.append(name)
        fig.savefig(name)  # Сохраняет текущее состояние.

    for name in filenames:  # Выводит все итерационные графики.
        print(f'<img width="900px" src="/resources/{name}">')

def main():
    print("Find:")  # Сообщение о начале поиска.
    interval = [-2, 1]  # Задаёт начальный интервал (сужен по сравнению с предыдущим примером).
    tol = 1e-2  # Задаёт относительно грубую точность остановки (0.01).
    [xmin, f, neval, coords] = isearch(interval, tol)  # Вызывает метод обратной параболической интерполяции.
    print([xmin, f, neval])  # Выводит результат: найденная точка, значение функции, число вычислений.
    opisearchslides(coords, interval)  # Вызывает функцию визуализации.

if __name__ == '__main__':
    main()  # Запускает основную функцию, если скрипт запущен напрямую.