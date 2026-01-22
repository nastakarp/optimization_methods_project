import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.
matplotlib.use('Agg')  # Устанавливает backend 'Agg' — позволяет сохранять графики без GUI (для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.
import random  # Импортирует модуль random для генерации случайных цветов при визуализации.
'''
def f(x): return x ** 2 - 10 * np.cos(0.3 * np.pi * x) - 20  # Определяет целевую функцию f(x) — осциллирующая квадратичная функция с множеством локальных минимумов.

def df(x): return 2 * x + 3 * np.pi * np.sin(0.3 * np.pi * x)  # Определяет аналитическую производную f'(x).
'''
#тут нужно поставить интервал (-2,9.9)
def f1(x):
    return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def df1(x):
    """
    Аналитическая производная f1(x).
    Не определена в x = -20 и x = 10 (особые точки из-за abs).
    Возвращает вещественное число; в особых точках — приближение.
    """
    eps = 1e-15  # порог для избежания деления на ноль

    # Часть 1: d/dx [ -2 * sin(sqrt(|x/2 + 10|)) ]
    u = x / 2 + 10
    abs_u = abs(u)
    if abs_u < eps:
        term1 = 0.0  # или np.nan — но 0 безопаснее для численных методов
    else:
        sign_u = np.sign(u)
        sqrt_abs_u = np.sqrt(abs_u)
        term1 = -2 * np.cos(sqrt_abs_u) * (sign_u / 4) / sqrt_abs_u
        # Обоснование: d/dx sqrt(|u|) = (1/(2*sqrt(|u|))) * sign(u) * du/dx,
        # где du/dx = 1/2 → итого: sign(u) / (4 * sqrt(|u|))

    # Часть 2: d/dx [ -x * sin(sqrt(|x - 10|)) ] = -sin(...) - x * cos(...) * d/dx[sqrt(|x-10|)]
    v = x - 10
    abs_v = abs(v)
    if abs_v < eps:
        # При v ≈ 0: sqrt(|v|) ≈ 0, sin(0)=0, cos(0)=1, но производная sqrt(|v|) → ∞
        # Однако множитель x * ... может компенсировать? Лучше использовать предел.
        # Практически: ставим 0 или игнорируем особую точку.
        term2 = -np.sin(0.0)  # = 0
        term3 = 0.0
    else:
        sign_v = np.sign(v)
        sqrt_abs_v = np.sqrt(abs_v)
        term2 = -np.sin(sqrt_abs_v)
        term3 = -x * np.cos(sqrt_abs_v) * (sign_v / (2 * sqrt_abs_v))

    return term1 + term2 + term3

#тут нужно поставить интервал (-2,10)
def f(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df(x):
    """
    Аналитическая производная f2(x) = x^2 - 10*cos(0.5*pi*x) - 110.
    f2'(x) = 2x + 5*pi*sin(0.5*pi*x)
    """
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)


def razn2(x0, x1):
    return (df(x1) - df(x0)) / (x1 - x0)  # Вычисляет первую разделённую разность (аналог первой производной) для двух точек.

def razn3(x0, x1, x2):
    return (razn2(x1, x2) - razn2(x0, x1)) / (x2 - x0)  # Вычисляет вторую разделённую разность (аналог второй производной) для трёх точек.

def w(xk, xk1, xk2):
    return razn2(xk1, xk) + razn2(xk2, xk) - razn2(xk2, xk1)  # Вспомогательная функция, используемая в формуле метода Мюллера.

def msearch(interval, tol):
    a, b = interval  # Распаковывает входной интервал [a, b].
    x0 = (a + b) / 2  # Берёт начальную точку как середину интервала.
    coords = [[x0, a, b]]  # Инициализирует список для хранения истории: [текущая точка, левая граница, правая граница].

    g = df(x0)  # Вычисляет производную в начальной точке.
    # Вычисляет коэффициенты квадратного уравнения для нахождения корня аппроксимирующей параболы:
    d1 = w(x0, b, a) + np.sqrt(w(x0, b, a) ** 2 - 4 * g * razn3(a, b, x0))
    d2 = w(x0, b, a) - np.sqrt(w(x0, b, a) ** 2 - 4 * g * razn3(a, b, x0))

    # Выбирает корень с большим модулем (для числовой устойчивости):
    if abs(d1) > abs(d2):
        xmin = x0 - (2 * g) / d1
    else:
        xmin = x0 - (2 * g) / d2

    # Обновляет интервал, исходя из положения нового приближения:
    if (xmin < x0):
        b = x0  # Новое приближение слева → сужаем справа.
    else:
        a = x0  # Новое приближение справа → сужаем слева.

    neval = 3  # Счётчик вычислений производной: df(x0), df(a), df(b) — всего 3.

    # Итерационный цикл:
    while abs(xmin - coords[-1][0]) > tol:  # Пока изменение точки больше допуска:
        coords.append([xmin, a, b])  # Сохраняем текущее состояние.
        xl = xmin  # Сохраняем предыдущую точку.
        g = df(xmin)  # Вычисляем производную в новой точке.

        # Повторяем вычисление корней квадратного уравнения:
        d1 = w(xmin, b, a) + np.sqrt(w(xmin, b, a) ** 2 - 4 * g * razn3(a, b, xmin))
        d2 = w(xmin, b, a) - np.sqrt(w(xmin, b, a) ** 2 - 4 * g * razn3(a, b, xmin))

        # Выбираем более устойчивый корень:
        if abs(d1) > abs(d2):
            xmin -= (2 * g) / d1
        else:
            xmin -= (2 * g) / d2

        # Обновляем интервал:
        if (xmin < xl):
            b = xl
        else:
            a = xl

        neval += 3  # Увеличиваем счётчик за три новых вычисления df (внутри w и razn3).

    fmin = f(xmin)  # Вычисляем значение функции в найденной точке минимума.

    answer_ = [xmin, fmin, neval, coords]  # Формируем результат.
    return answer_  # Возвращает результат.

# ---------- Базовые графики ----------
def drawdf(a, b, h, ax1):
    x = np.arange(a, b, h)  # Создаёт массив x от a до b с шагом h.
    y = [df(i) for i in x]  # Вычисляет значения производной f'(x).
    ax1.plot(x, y, lw=1, c='black')  # Рисует график f'(x) чёрной линией.
    ax1.plot([a, b], [0, 0], c='black', lw=1)  # Горизонтальная линия y=0.
    ax1.set_xlabel('x')  # Подписывает ось X.
    ax1.set_ylabel("f'(x)")  # Подписывает ось Y как производную.
    ax1.set_xlim([a - 0.1, b + 0.1])  # Устанавливает пределы по X.

def drawf(a, b, h, ax2):
    x = np.arange(a, b, h)  # Массив x.
    y = [f(i) for i in x]  # Значения функции f(x).
    ax2.plot(x, y, lw=1, c='black')  # График f(x).
    ax2.plot([a, b], [0, 0], c='black', lw=1)  # Линия y=0.
    ax2.set_xlabel('x')  # Подпись X.
    ax2.set_ylabel("f(x)")  # Подпись Y.
    ax2.set_xlim([a - 0.1, b + 0.1])  # Пределы по X.

# ---------- Итерационные точки и парабола для df ----------
def drawpointsDF(coord, ax1, i):
    xk, a, b = coord  # Распаковывает текущую точку и границы интервала.
    color = (random.random(), random.random(), random.random())  # Случайный цвет.

    # Текущая точка (аппроксимированный корень f'(x)=0):
    ax1.plot(xk, df(xk), marker='*', ms=10, c=color)
    ax1.text(xk, df(xk) + 2, str(i + 1))  # Номер итерации над точкой.

    # Границы интервала:
    ax1.scatter([a, b], [df(a), df(b)], marker='o', c=color)

    # Вертикальная линия от точки до оси X:
    ax1.plot([xk, xk], [df(xk), 0], c=color, lw=1)

    # ---------- Парабола Мюллера для df ----------
    x_vals = np.array([xk, a, b])  # Три точки для интерполяции.
    y_vals = np.array([df(xk), df(a), df(b)])  # Их значения производной.
    coeffs = np.polyfit(x_vals, y_vals, 2)  # Находит коэффициенты квадратного полинома.

    t = np.linspace(a, b, 100)  # Плотная сетка для плавной параболы.
    parabola = np.polyval(coeffs, t)  # Вычисляет значения параболы.
    ax1.plot(t, parabola, lw=2, c=color, ls='--')  # Рисует пунктирную параболу.

def drawpointsF(coord, ax2, i):
    xk, a, b = coord
    color = (random.random(), random.random(), random.random())

    # Отмечает текущую точку на графике f(x):
    ax2.plot(xk, f(xk), marker='*', ms=10, c=color)
    ax2.text(xk, f(xk) + 1.2, str(i + 1))  # Номер итерации.
    ax2.scatter([a, b], [f(a), f(b)], marker='o', c=color)  # Границы интервала.

# ---------- Основная функция визуализации ----------
def mullersearchslides(coords, interval):
    fig = plt.figure(figsize=(10, 8))  # Создаёт фигуру размером 10×8 дюймов.
    ax1 = fig.add_subplot(2, 1, 1)  # Верхний график — для f'(x).
    ax2 = fig.add_subplot(2, 1, 2)  # Нижний график — для f(x).
    fig.suptitle('Muller method visualisation')  # Общий заголовок.

    a, b = interval  # Исходный интервал.
    h = (b - a) / 200  # Шаг дискретизации.

    drawdf(a, b, h, ax1)  # Рисует базовый график производной.
    drawf(a, b, h, ax2)  # Рисует базовый график функции.

    print("F(x) and F'(x):")  # Выводит заголовок.
    fig.savefig("plot.png")  # Сохраняет базовый график.
    print('<img width="900px" src="/resources/plot.png">')  # HTML-тег для отображения.

    print("Steps:")  # Заголовок перед итерациями.
    nfigs = min(10, len(coords))  # Ограничивает число визуализируемых итераций до 10.
    filenames = []  # Список имён файлов.

    for i in range(nfigs):  # Цикл по итерациям.
        drawpointsDF(coords[i], ax1, i)  # Добавляет точки и параболу на график f'(x).
        drawpointsF(coords[i], ax2, i)  # Добавляет точки на график f(x).

        ax1.set_title(f"Iteration {i + 1}")  # Заголовок верхнего графика.
        name = f"iter{i + 1}.png"
        filenames.append(name)
        fig.savefig(name)  # Сохраняет текущее состояние.

    for name in filenames:  # Выводит все итерационные графики.
        print(f'<img width="900px" src="/resources/{name}">')

def main():
    print("Find:")  # Сообщение о начале поиска.
    interval = [-2, 10]  # Задаёт начальный интервал.
    tol = 1e-5  # Задаёт точность остановки.
    [xmin, f, neval, coords] = msearch(interval, tol)  # Вызывает метод Мюллера.
    print([xmin, f, neval])  # Выводит результат.
    mullersearchslides(coords, interval)  # Вызывает визуализацию.

if __name__ == '__main__':
    main()  # Запускает основную функцию, если скрипт запущен напрямую.