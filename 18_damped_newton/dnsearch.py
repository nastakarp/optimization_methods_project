import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI (для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt.
import random  # Импортирует модуль random для генерации случайных цветов.

def f(x): return x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20  # Определяет целевую функцию f(x) — осциллирующая квадратичная функция.

def df(x): return 2 * x + 3 * np.pi * np.sin(0.3 * np.pi * x)  # Определяет первую производную f'(x).

def ddf(x): return 2 + 0.9 * (np.pi ** 2) * np.cos(0.3 * np.pi * x)  # Определяет вторую производную f''(x).

def nsearch(tol, x0):
    # Определяет функцию nsearch для поиска минимума методом, сочетающим Ньютона и идеи Нестерова.
    #   Вход: tol — точность остановки, x0 — начальная точка.
    #   Выход: [xmin, fmin, neval, coords]

    x = x0  # Инициализирует текущую точку как начальную.
    coords = [x]  # Инициализирует список для хранения траектории поиска (включая начальную точку).
    neval = 0  # Счётчик вычислений производных (градиента и гессиана).

    while True:  # Бесконечный цикл — выход по условию внутри.
        g = df(x)  # Вычисляет первую производную (градиент) в текущей точке x.
        h = ddf(x)  # Вычисляет вторую производную (гессиан, в 1D — просто f''(x)).
        neval += 2  # Увеличивает счётчик за два вычисления: df и ddf.

        # декремент delta = g * H^-1 * g
        delta = (g ** 2) / h  # Вычисляет "декремент" — квадратичную форму g²/h, аналогичную ||∇f||² в метрике Гессе.

        alpha = 1 / (1 + delta)  # переменный шаг (Нестеров)

        x_new = x - alpha * (g / h)  # переход в новую точку

        if abs(x_new - x) < tol:  # Если изменение позиции меньше допуска — завершаем поиск.
            x = x_new  # Обновляем текущую точку.
            coords.append(x)  # Сохраняем финальную точку в траекторию.
            break  # Выходим из цикла.

        x = x_new  # Иначе обновляем текущую точку.
        coords.append(x)  # Добавляем новую точку в траекторию.

    return [x, f(x), neval + 1, coords]  # Возвращает результат:
    # x — найденный минимум,
    # f(x) — значение функции в минимуме,
    # neval + 1 — общее число вычислений (добавлено одно для f(x)),
    # coords — полная траектория поиска.


def drawdf(a, b, h, ax1):
    color = (random.random(), random.random(), random.random())  # Случайный цвет для графика.
    x_ = np.arange(a, b + h, h)  # Массив x от a до b с шагом h.
    y = [df(i) for i in x_]  # Значения первой производной.

    ax1.plot(x_, y, lw=1, c=color)  # Рисует график f'(x).
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)  # Отмечает границы интервала.

    ax1.set_xlabel('x')  # Подписывает ось X.
    ax1.set_ylabel("f'(x)")  # Подписывает ось Y.

    ax1.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)  # Горизонтальная линия y=0.
    ax1.set_xlim([-2.1, 7.1])  # Фиксирует пределы по X.
    ax1.set_ylim([-30, 40])  # Фиксирует пределы по Y.

def drawf(a, b, h, ax2):
    color = (random.random(), random.random(), random.random())  # Случайный цвет.
    x_ = np.arange(a, b + h, h)  # Массив x.
    y = [f(i) for i in x_]  # Значения функции f(x).

    ax2.plot(x_, y, lw=1, c=color)  # График f(x).
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)  # Границы интервала.

    ax2.set_xlabel('x')
    ax2.set_ylabel("f(x)")

    ax2.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)  # Линия y=0.
    ax2.set_xlim([-2.1, 7.1])  # Пределы по X.

def drawpointsDF(coord, ax1, a, b, h, i):
    # Отмечает текущую точку на графике f'(x)
    ax1.plot(coord, df(coord), marker='*', ms=10)  # Звезда в точке (x, f'(x)).
    ax1.text(coord, df(coord) + 4, str(i + 1), backgroundcolor='white')  # Номер итерации.

    # Рисует касательную к f'(x) в точке coord (это прямая с наклоном f''(x))
    x_ = np.arange(a, b + h, h)  # Массив для построения касательной.
    new_color = (random.random(), random.random(), random.random())  # Новый случайный цвет.
    k = ddf(coord)  # Наклон касательной = f''(x).
    br = df(coord)  # Значение f'(x) в точке.
    yt = k * (x_ - coord) + br  # Уравнение касательной: y = k*(x - x0) + y0.
    ax1.plot(x_, yt, c=new_color, lw=1)  # Рисует касательную.
    ax1.plot([coord, coord], [0, df(coord)], c=(0, 0, 0), lw=0.5)  # Вертикальная линия до оси X.

def drawpointsF(coord, ax2, i):
    # Отмечает текущую точку на графике f(x)
    ax2.plot(coord, f(coord), marker='*', ms=10)
    ax2.text(coord, f(coord) + 4, str(i + 1), backgroundcolor='white')  # Номер итерации.

def newtondrawfig(interval, coord):
    # draw graphics
    fig = plt.figure(figsize=[20, 10])  # Создаёт большую фигуру.
    ax1 = fig.add_subplot(2, 1, 1)  # Верхний график — для f'(x).
    ax2 = fig.add_subplot(2, 1, 2)  # Нижний график — для f(x).
    fig.suptitle('Newton visualisation')  # Общий заголовок.

    a = interval[0]  # Левая граница для визуализации.
    b = interval[1]  # Правая граница.
    h = (b - a) / 100  # Шаг дискретизации.

    # Рисует базовые графики
    drawdf(a, b, h, ax1)
    drawf(a, b, h, ax2)
    print("F(x) and F'(x):")
    name = "plot.png"
    fig.savefig(name)
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"
    print(ad)

    DictName = []  # Список имён файлов.
    print("Steps:")
    nfigs = min([10, len(coord)])  # Ограничивает число итераций до 10.
    for i in range(nfigs):
        drawpointsDF(coord[i], ax1, a, b, h, i)  # Добавляет точку и касательную на f'(x).
        drawpointsF(coord[i], ax2, i)  # Добавляет точку на f(x).

        ax1.set_title(str(i + 1) + " " + 'Iteration')  # Заголовок верхнего графика.
        DictName.append("iter" + str(i + 1) + ".png")
        fig.savefig(DictName[i])  # Сохраняет график.

    for elem in DictName:  # Выводит все итерационные графики.
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"
        print(ad)

def main():
    print("Find:")
    interval = [-2, 7]  # Интервал для визуализации (не влияет на поиск!).
    tol = 0.01  # Точность остановки.
    [xmin, f, neval, coords] = nsearch(tol, 1.3)  # Запускает метод Ньютона с начальной точкой 1.3.
    print([xmin, f, neval])  # Выводит результат.
    newtondrawfig(interval, coords)  # Визуализирует процесс.

if __name__ == '__main__':
    main()  # Запускает основную функцию.