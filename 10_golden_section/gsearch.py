import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики в файлы без отображения GUI (подходит для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.
import random  # Импортирует модуль random для генерации случайных цветов при визуализации.

def f1(x):     return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def f2(x):    return x**2-10*np.cos(0.5* np.pi*x)-110

def f(x):     return x**2-10*np.cos(0.5* np.pi*x)-110  # Определяет целевую функцию f(x); можно упростить: f(x) = 2x² - 9x - 31 (та же, что и раньше).

def gsearch(interval, tol):
    # GOLDENSECTIONSEARCH searches for minimum using golden section
    # 	[xmin, fmin, neval] = GOLDENSECTIONSEARCH(f,interval,tol)
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
    phi = (np.sqrt(5) - 1) / 2  # Вычисляет константу золотого сечения φ ≈ 0.618.
    coord = []  # Инициализирует список для хранения истории точек на каждой итерации.
    neval = 0  # Счётчик количества вычислений функции f(x).

    x1 = b - phi * (b - a)  # Вычисляет первую внутреннюю точку x1 по правилу золотого сечения.
    x2 = a + phi * (b - a)  # Вычисляет вторую внутреннюю точку x2 (симметрично x1).
    f1 = f(x1)  # Вычисляет значение функции в x1.
    f2 = f(x2)  # Вычисляет значение функции в x2.
    neval += 2  # Увеличивает счётчик вычислений на 2 (за f1 и f2).
    coord.append([x1, x2, a, b])  # Сохраняет начальные точки и границы интервала.

    while np.abs(b - a) > tol:  # Пока длина интервала больше заданной точности:
        if f1 < f2:  # Если f(x1) < f(x2), минимум находится в левой части [a, x2].
            b = x2  # Сужаем интервал до [a, x2].
            x2 = x1  # Переносим x1 → x2.
            f2 = f1  # Значение f1 теперь становится f2.
            x1 = b - phi * (b - a)  # Вычисляем новую точку x1.
            f1 = f(x1)  # Вычисляем f(x1).
        else:  # Иначе минимум в правой части [x1, b].
            a = x1  # Сужаем интервал до [x1, b].
            x1 = x2  # Переносим x2 → x1.
            f1 = f2  # Значение f2 теперь становится f1.
            x2 = a + phi * (b - a)  # Вычисляем новую точку x2.
            f2 = f(x2)  # Вычисляем f(x2).
        neval += 1  # Увеличиваем счётчик за одно новое вычисление f(x) (либо f1, либо f2).
        coord.append([x1, x2, a, b])  # Сохраняем текущее состояние интервала и точек.

    if f1 < f2:  # После цикла выбираем лучшую из двух оставшихся точек.
        xmin = x1
        fmin = f1
    else:
        xmin = x2
        fmin = f2

    answer_ = [xmin, fmin, neval, coord]  # Формируем результат: минимум, значение, число вычислений, история.
    return answer_  # Возвращает результат.

def drawplot(a, b, x1, x2, ax, color):
    h = (b - a) / 100  # Шаг дискретизации для плавного графика на отрезке [a, b].
    x = np.arange(a, b + h, h)  # Создаёт массив значений x от a до b с шагом h.
    y = [f(i) for i in x]  # Вычисляет значения функции f(x) для каждого x.
    ax.plot(x, y, lw=1, c=color)  # Рисует график функции тонкой линией заданного цвета.
    ax.scatter([a, b], [f(a), f(b)], marker='o', c=color)  # Отмечает границы интервала [a, b] кружками.
    ax.set_xlabel('x')  # Подписывает ось X.
    ax.set_ylabel('y')  # Подписывает ось Y.
    ax.plot([a, b], [0, 0], c=(0, 0, 0), lw=1.2)  # Рисует горизонтальную линию y=0 чёрным цветом.

    y1 = f(x1)  # Значение функции в x1.
    ax.plot([x1, x1], [0, y1], lw=1, c=color, marker='s', ms=4)  # Вертикальная линия от 0 до f(x1) с квадратным маркером.
    y2 = f(x2)  # Значение функции в x2.
    ax.plot([x2, x2], [0, y2], lw=1, c=color, marker='s', ms=4)  # Вертикальная линия от 0 до f(x2) с квадратным маркером.
    ax.set_xlim([-2.25, 10.25])  # Фиксирует пределы по оси X для согласованности всех графиков.

def placelabel(x, y, deltaX, deltaY, iternumber, ax):
    ax.text(x - deltaX / 2, y + 4 * deltaY, str(iternumber), backgroundcolor='white')  # Размещает надпись с номером итерации над указанной точкой с белым фоном.

def gs2slides(interval, coord):
    # data for visualisation
    a = interval[0]  # Левая граница исходного интервала.
    b = interval[1]  # Правая граница исходного интервала.
    h = (b - a) / 100  # Шаг для построения графика.
    x = np.arange(a, b + 0.2, h)  # Массив x для полного графика (с небольшим запасом).
    y = [f(i) for i in x]  # Значения функции на этом массиве.
    miny = min(y)  # Минимальное значение функции на интервале.
    maxy = max(y)  # Максимальное значение функции на интервале.
    deltaX = (b - a) / 200  # Горизонтальный масштаб для размещения меток.
    deltaY = abs(maxy - miny) / 200  # Вертикальный масштаб для размещения меток.
    Nfigs = 10  # Максимальное количество итераций для визуализации (можно изменить).

    # plot
    fig, ax = plt.subplots(figsize=[30, 13])  # Создаёт большую фигуру для детальной визуализации.
    fig.suptitle('Golden section visualisation')  # Устанавливает общий заголовок фигуры.
    placelabel(a, 0, deltaX, deltaY, 1, ax)  # Метка "1" над левой границей.
    placelabel(b, 0, deltaX, deltaY, 1, ax)  # Метка "1" над правой границей.
    color = (random.random(), random.random(), random.random())  # Генерирует случайный RGB-цвет.
    drawplot(a, b, coord[0][0], coord[0][1], ax, color)  # Рисует начальное состояние (первая итерация).
    print("F(x) and F'(x):")  # Выводит заголовок (хотя производная не используется — возможно, опечатка).
    name = "plot.png"  # Имя файла для базового графика.
    fig.savefig(name)  # Сохраняет график.
    ad = "<img width=\"1500px\" src=\"/resources/" + name + "\">"  # Формирует HTML-тег для отображения.
    print(ad)  # Выводит HTML-тег.

    DictName = []  # Список имён файлов для последующего вывода.
    print("Steps:")  # Выводит заголовок перед серией итераций.

    nfigs = min([Nfigs, len(coord)])  # Ограничивает число визуализируемых итераций до Nfigs или меньше.

    j = 0  # Счётчик реально отрисованных итераций.
    for i in range(nfigs):  # Цикл по выбранным итерациям.
        color = (random.random(), random.random(), random.random())  # Новый случайный цвет для каждой итерации.

        drawplot(coord[i][2], coord[i][3], coord[i][0], coord[i][1], ax, color)  # Рисует текущий интервал и точки.

        placelabel(coord[i][0], 0, deltaX, deltaY, i + 1, ax)  # Метка номера итерации над x1.
        placelabel(coord[i][1], 0, deltaX, deltaY, i + 1, ax)  # Метка номера итерации над x2.

        ax.set_title(str(i + 1) + " " + 'Iteration')  # Устанавливает заголовок осей как "N Iteration".
        DictName.append("iter" + str(i + 1) + ".png")  # Добавляет имя файла в список.
        fig.savefig(DictName[i])  # Сохраняет текущее состояние графика.
        j += 1  # Увеличивает счётчик отрисованных итераций.

    ax.set_title(str(j + 1) + " " + 'Iteration')  # Устанавливает заголовок для следующей (финальной) итерации.
    DictName.append("iter" + str(j + 1) + ".png")  # Добавляет ещё один файл (возможно, лишний — зависит от логики).
    fig.savefig(DictName[j])  # Сохраняет его.

    for elem in DictName:  # Цикл по всем сохранённым графикам.
        ad = "<img width=\"1500px\" src=\"/resources/" + elem + "\">"  # Формирует HTML-тег.
        print(ad)  # Выводит HTML-тег для отображения в веб-интерфейсе.

# Основной исполняемый блок:
print("Find:")  # Выводит сообщение о начале поиска.
interval = [-2, 10]  # Задаёт начальный интервал поиска минимума.
tol = 1e-10  # Задаёт высокую точность остановки.
[xmin, fmin, neval, coords] = gsearch(interval, tol)  # Вызывает метод золотого сечения.
print([xmin, fmin, neval])  # Выводит найденный минимум, значение функции и число вычислений.

gs2slides(interval, coords)  # Вызывает функцию визуализации процесса поиска.

'''
print("\nИтерации метода золотого сечения:")
print(f"{'Шаг':>3} | {'a':>10} {'b':>10} | {'x1':>10} {'x2':>10} | {'L = b-a':>12}")
print("-" * 60)
for i, (x1, x2, a, b) in enumerate(coords):
    L = b - a
    print(f"{i:3d} | {a:10.6f} {b:10.6f} | {x1:10.6f} {x2:10.6f} | {L:12.6e}")
'''
# Этот блок закомментирован — он предназначен для текстового вывода таблицы итераций в консоль.