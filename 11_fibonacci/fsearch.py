import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики в файлы без GUI (подходит для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.
import random  # Импортирует модуль random для генерации случайных цветов при визуализации.

def f(x):     return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def f2(x):    return x**2-10*np.cos(0.5* np.pi*x)-110


def makefib(n):
    fib = [1, 1]  # Инициализирует список первых двух чисел Фибоначчи.
    for i in range(n - 2):  # Добавляет оставшиеся (n-2) чисел.
        fib.append(fib[-1] + fib[-2])  # Каждое следующее число — сумма двух предыдущих.
    return fib  # Возвращает список первых n чисел Фибоначчи.

def findfib(num):
    f0, f1 = 1, 1  # Начальные значения последовательности Фибоначчи.
    f = f0 + f1  # Третье число: 1 + 1 = 2.
    n = 3  # Счётчик текущего номера в последовательности.
    while f < num:  # Пока текущее число Фибоначчи меньше заданного num:
        prev = f  # Сохраняем текущее значение как предыдущее.
        f = f + f1  # Вычисляем следующее число: f_new = f + f1.
        f1 = prev  # Обновляем f1 на предыдущее значение f.
        n += 1  # Увеличиваем номер.
    return n  # Возвращает наименьший номер n, для которого F_n >= num.

def fsearch(interval, tol):
    # Реализует метод поиска минимума с использованием последовательности Фибоначчи.
    #   INPUT ARGUMENTS
    # 	f is a function
    # 	interval = [a, b] - search interval
    # 	tol - set for bot range and function value
    #   OUTPUT ARGUMENTS
    # 	xmin is a function minimizer
    # 	fmin = f(xmin)
    # 	neval - number of function evaluations
    #   coords - array of statistics,  coord[i][:] =  [x1,x2, a, b]

    a, b = interval  # Распаковывает входной интервал [a, b].
    n = findfib((b - a) / tol)  # Находит минимальное n, такое что F_n >= (b-a)/tol — определяет количество итераций.
    fib = makefib(n)  # Генерирует первые n чисел Фибоначчи.
    n -= 1  # Корректирует n для удобства индексации (теперь n = исходное n - 1).

    x1 = a + (fib[n - 2] / fib[n]) * (b - a)  # Вычисляет левую внутреннюю точку x1 по формуле Фибоначчи.
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)  # Вычисляет правую внутреннюю точку x2.
    print(a, b, fib[n-2], fib[n-1], fib[n], n)  # Отладочный вывод: границы и используемые числа Фибоначчи.

    coord = []  # Инициализирует список для хранения истории точек и интервалов.
    neval = 1  # Счётчик вычислений функции; будет обновляться в цикле.

    while True:  # Бесконечный цикл — выход по условию внутри.
        coord.append([x1, x2, a, b])  # Сохраняет текущие точки и границы интервала.

        if f(x1) < f(x2):  # Если f(x1) < f(x2), минимум слева → сужаем интервал до [a, x2].
            b = x2  # Новая правая граница — x2.
            x2 = x1  # Переносим x1 → x2.
            # Вычисляем новую точку x1 с использованием чисел Фибоначчи:
            x1 = a + fib[n - neval - 2] / fib[n - neval] * (b - a)
        elif f(x1) >= f(x2):  # Иначе минимум справа → сужаем до [x1, b].
            a = x1  # Новая левая граница — x1.
            x1 = x2  # Переносим x2 → x1.
            # Вычисляем новую точку x2:
            x2 = a + fib[n - neval - 1] / fib[n - neval] * (b - a)
        neval += 1  # Увеличиваем счётчик вычислений.

        if neval == n - 1:  # Если достигли предпоследней итерации:
            x2 = x1 + tol  # Создаём финальную точку чуть правее x1 (на расстоянии tol).
            if f(x1) > f(x2):  # Сравниваем f(x1) и f(x2) для выбора финального интервала.
                xmin = (x1 + b) / 2  # Если f(x2) лучше — минимум в [x1, b].
            else:
                xmin = (x2 + a) / 2  # Иначе — в [a, x2].
            break  # Выходим из цикла.

    fmin = f(xmin)  # Вычисляет значение функции в найденной точке минимума.

    answer_ = [xmin, fmin, neval, coord]  # Формирует результат.
    return answer_  # Возвращает результат.

def drawplot(a, b, x1, x2, ax, color):
    h = (b - a) / 100  # Шаг дискретизации для плавного графика.
    x = np.arange(a, b + h, h)  # Создаёт массив x от a до b с шагом h.
    y = [f(i) for i in x]  # Вычисляет значения функции на этом массиве.
    ax.plot(x, y, lw=1, c=color)  # Рисует график функции тонкой линией заданного цвета.
    ax.scatter([a, b], [f(a), f(b)], marker='o', c=color)  # Отмечает границы интервала кружками.
    ax.set_xlabel('x')  # Подписывает ось X.
    ax.set_ylabel('y')  # Подписывает ось Y.
    ax.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)  # Рисует горизонтальную линию y=0 чёрным цветом.

    y1 = f(x1)  # Значение функции в x1.
    ax.plot([x1, x1], [0, y1], lw=1, c=color, marker='s', ms=4)  # Вертикальная линия до f(x1) с квадратным маркером.
    y2 = f(x2)  # Значение функции в x2.
    ax.plot([x2, x2], [0, y2], lw=1, c=color, marker='s', ms=4)  # Вертикальная линия до f(x2) с квадратным маркером.
    ax.set_xlim([-2.25, 10.25])  # Фиксирует пределы по оси X для всех графиков.

def placelabel(x, y, deltaX, deltaY, iternumber, ax):
    ax.text(x - deltaX / 2, y + 4 * deltaY, str(iternumber), backgroundcolor='white')  # Размещает надпись с номером итерации над точкой.

def fs2slides(interval, coord):
    # data for visualisation
    a = interval[0]  # Левая граница исходного интервала.
    b = interval[1]  # Правая граница исходного интервала.
    h = (b - a) / 100  # Шаг для построения графика.
    x = np.arange(a, b + 0.2, h)  # Массив x для полного графика (с небольшим запасом).
    y = [f(i) for i in x]  # Значения функции на этом массиве.
    miny = min(y)  # Минимальное значение функции.
    maxy = max(y)  # Максимальное значение функции.
    deltaX = (b - a) / 200  # Горизонтальный масштаб для меток.
    deltaY = abs(maxy - miny) / 200  # Вертикальный масштаб для меток.
    Nfigs = 10  # Максимальное число итераций для визуализации.

    # plot
    fig, ax = plt.subplots(figsize=[30, 13])  # Создаёт большую фигуру.
    fig.suptitle('Fibonacci visualisation')  # Общий заголовок фигуры.
    placelabel(a, 0, deltaX, deltaY, 1, ax)  # Метка "1" над левой границей.
    placelabel(b, 0, deltaX, deltaY, 1, ax)  # Метка "1" над правой границей.
    color = (random.random(), random.random(), random.random())  # Случайный цвет.
    drawplot(a, b, coord[0][0], coord[0][1], ax, color)  # Рисует начальное состояние.
    print("F(x) and F'(x):")  # Выводит заголовок (производная не используется — возможно, шаблон от другого метода).
    name = "plot.png"  # Имя файла для базового графика.
    fig.savefig(name)  # Сохраняет график.
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"  # HTML-тег для отображения.
    print(ad)  # Выводит тег.

    DictName = []  # Список имён файлов.
    print("Steps:")  # Заголовок перед серией итераций.

    nfigs = min([Nfigs, len(coord)])  # Ограничивает число визуализируемых итераций.

    j = 0  # Счётчик реально отрисованных итераций.
    for i in range(nfigs):  # Цикл по выбранным итерациям.
        color = (random.random(), random.random(), random.random())  # Новый случайный цвет.

        drawplot(coord[i][2], coord[i][3], coord[i][0], coord[i][1], ax, color)  # Рисует текущий интервал и точки.

        placelabel(coord[i][0], 0, deltaX, deltaY, i + 1, ax)  # Метка над x1.
        placelabel(coord[i][1], 0, deltaX, deltaY, i + 1, ax)  # Метка над x2.

        ax.set_title(str(i + 1) + " " + 'Iteration')  # Заголовок осей.
        DictName.append("iter" + str(i + 1) + ".png")  # Добавляет имя файла.
        fig.savefig(DictName[i])  # Сохраняет график.
        j += 1  # Увеличивает счётчик.

    ax.set_title(str(j + 1) + " " + 'Iteration')  # Заголовок для следующей итерации.
    DictName.append("iter" + str(j + 1) + ".png")  # Добавляет ещё один файл.
    fig.savefig(DictName[j])  # Сохраняет его.

    for elem in DictName:  # Цикл по всем графикам.
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"  # Формирует HTML-тег.
        print(ad)  # Выводит тег.

# Основной исполняемый блок:
print("Find:")  # Выводит сообщение о начале поиска.
interval = [-2, 10]  # Задаёт начальный интервал поиска.
tol = 1e-5  # Задаёт допуск (точность); здесь довольно грубый — 0.5.
[xmin, fmin, neval, coords] = fsearch(interval, tol)  # Вызывает метод Фибоначчи.
print([xmin, fmin, neval])  # Выводит результат: минимум, значение функции, число вычислений.

fs2slides(interval, coords)  # Вызывает функцию визуализации процесса поиска.