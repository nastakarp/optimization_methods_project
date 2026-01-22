import numpy as np  # Импортирует библиотеку NumPy для числовых вычислений и работы с массивами.
import numpy as np  # Повторный импорт (избыточен, но безвреден — можно удалить одну из строк).
import matplotlib  # Импортирует основной модуль Matplotlib для визуализации данных.

matplotlib.use('Agg')  # Устанавливает backend 'Agg' для сохранения графиков в файлы без GUI (подходит для серверов/headless-режима).
import matplotlib.pyplot as plt  # Импортирует pyplot из Matplotlib под псевдонимом plt — основной инструмент построения графиков.

def f(x): return 2 * (x ** 2) - 9 * x - 31  # Определяет целевую функцию f(x) = 2x² - 9x - 31.

#def df(x): return 4 * x - 9  # Определяет производную f'(x) = 4x - 9 (не используется в этом алгоритме, но оставлена для совместимости).

def tpsearch(interval, tol):  # Функция поиска минимума методом трёхточечного поиска (похож на метод золотого сечения или трисекции).
    # searches for minimum using bisection method
    # arguments: bisectionsearch(f,df,interval,tol)
    # f - an objective function
    # df -  an objective function derivative
    # interval = [a, b] - search interval
    # tol - tolerance for both range and function value
    # output: [xmin, fmin, neval, coords]
    # xmin - value of x in fmin
    # fmin - minimul value of f
    # neval - number of function evaluations
    # coords - array of x values found during optimization

    coords = []  # Инициализирует список для хранения записей о точках на каждой итерации.

    a, b = interval  # Распаковывает входной интервал [a, b].
    xm = (a + b) / 2  # Вычисляет начальную среднюю точку xm.
    neval = 1  # Счётчик вычислений функции; здесь он инициализирован как 1, но на самом деле будет увеличиваться позже (ошибка логики — см. ниже).

    while abs(b - a) > tol:  # Пока длина интервала больше заданной точности:
        x1 = a + abs(b - a) / 4  # Вычисляет первую внутреннюю точку — на 1/4 отрезка от a.
        x2 = b - abs(b - a) / 4  # Вычисляет вторую внутреннюю точку — на 1/4 отрезка от b (симметрично x1).

        coords.append([x1, xm, x2, a, b])  # Сохраняет текущие точки и границы интервала для визуализации.

        if f(x1) < f(xm):  # Если значение в x1 меньше, чем в xm → минимум левее xm.
            b = xm  # Сужаем интервал до [a, xm].
            xm = x1  # Новая средняя точка — x1.
        elif f(x1) >= f(xm) and f(xm) <= f(x2):  # Если xm — локальный минимум между x1 и x2.
            a = x1  # Сужаем интервал до [x1, x2].
            b = x2
        else:  # Иначе минимум правее xm.
            a = xm  # Сужаем интервал до [xm, b].
            xm = x2  # Новая средняя точка — x2.
        xmin = xm  # Обновляем текущее приближение минимума.

    fmin = f(xmin)  # После выхода из цикла вычисляем значение функции в найденной точке минимума.

    answer_ = [xmin, fmin, neval, coords]  # Формируем результат: координата минимума, значение функции, число вычислений, история точек.
    return answer_  # Возвращает результат.

def trisearch2slides(interval, coords):  # Функция для визуализации процесса поиска минимума.
    t = np.arange(interval[0], interval[1], 0.001)  # Создаёт плотную сетку значений x для плавного графика.

    fig, ax = plt.subplots()  # Создаёт новую фигуру и оси.
    ax.plot(t, f(t), color='black', linewidth=1)  # Рисует исходную функцию тонкой чёрной линией.
    ax.set_xlabel('x')  # Подписывает ось X.
    ax.set_ylabel('f(x)')  # Подписывает ось Y.

    fig.savefig('plot.png')  # Сохраняет базовый график функции.
    print('<img width="550px" src="/resources/plot.png">')  # Выводит HTML-тег для отображения базового графика.

    nfigs = min(10, len(coords))  # Ограничивает количество выводимых итераций до 10 или меньше, если их мало.
    colors = plt.cm.tab10.colors  # Получает палитру из 10 цветов для различения итераций.

    for i in range(nfigs):  # Цикл по первым nfigs итерациям.
        x1, xm, x2, a, b = coords[i]  # Распаковывает данные текущей итерации.
        color = colors[i % len(colors)]  # Выбирает цвет из палитры (с зацикливанием, если итераций > 10).

        # дуга функции на текущем интервале [a, b]
        mask = (t >= a) & (t <= b)  # Создаёт булеву маску для выделения участка графика внутри [a, b].
        ax.plot(t[mask], f(t[mask]), color=color, linewidth=2)  # Рисует этот участок жирной цветной линией.

        # три точки одной итерации
        ax.plot([x1, xm, x2], [f(x1), f(xm), f(x2)], 'o', color=color)  # Отмечает три точки текущей итерации кружками.

        name = f'plot{i}.png'  # Формирует имя файла для i-го графика.
        fig.savefig(name)  # Сохраняет текущее состояние графика.
        print(f'<img width="550px" src="/resources/{name}">')  # Выводит HTML-тег для этого графика.

    # финальная итерация — подчёркиваем
    x1, xm, x2, a, b = coords[-1]  # Берём данные последней итерации.
    mask = (t >= a) & (t <= b)  # Маска для финального интервала.

    ax.plot(t[mask], f(t[mask]), color='black', linewidth=3)  # Рисует финальный интервал очень жирной чёрной линией.
    ax.plot([x1, xm, x2], [f(x1), f(xm), f(x2)], 'ko', markersize=8)  # Отмечает финальные точки крупными чёрными кружками.

    fig.savefig('plotFin.png')  # Сохраняет финальный график.
    print('Final:')  # Выводит заголовок перед финальным графиком.
    print('<img width="550px" src="/resources/plotFin.png">')  # Выводит HTML-тег для финального графика.

def main():  # Основная функция программы.
    print("Find:")  # Выводит сообщение о начале поиска.
    interval = [-2, 10]  # Задаёт начальный интервал поиска.
    tol = 1e-10  # Задаёт высокую точность остановки.
    [xmin, f, neval, coords] = tpsearch(interval, tol)  # Вызывает функцию поиска минимума.
    print([xmin, f, neval])  # Выводит найденный минимум, значение функции и число вычислений.
    trisearch2slides(interval, coords)  # Вызывает функцию визуализации.

if __name__ == '__main__':  # Проверяет, запущен ли скрипт напрямую.
    main()  # Если да — запускает основную функцию.