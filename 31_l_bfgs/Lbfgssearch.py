from numpy.linalg import norm                      # Импортирует функцию для вычисления евклидовой нормы вектора.
import numpy as np                                # Импортирует библиотеку NumPy для числовых операций.
import matplotlib                                 # Импортирует основной модуль Matplotlib для визуализации.
matplotlib.use('Agg')                             # Устанавливает backend 'Agg' для сохранения графиков без GUI.
import matplotlib.pyplot as plt                   # Импортирует pyplot для построения графиков.

def fSphere(X):
    if not isinstance(X, np.ndarray):             # Если X не массив NumPy — преобразуем.
        X = np.array(X)
    if X.shape == (2, 1):                         # Если X — столбец (2×1), делаем плоский вектор.
        X = X.flatten()
    return np.sum(X ** 2)                         # Возвращает сумму квадратов компонент (сферическая функция).

def dfSphere(X):
    if not isinstance(X, np.ndarray):             # Преобразуем вход в массив, если нужно.
        X = np.array(X)
    return 2 * X                                  # Градиент сферической функции: ∇f(x) = 2x.

def fLevy(X):
    """Levy function for d=2."""
    if not isinstance(X, np.ndarray):             # Приводим к массиву.
        X = np.array(X)
    if X.shape == (2, 1):                         # Преобразуем столбец в вектор.
        X = X.flatten()
    w1 = 1 + (X[0] - 1) / 4.0                     # Промежуточная переменная w1.
    w2 = 1 + (X[1] - 1) / 4.0                     # Промежуточная переменная w2.
    term1 = np.sin(np.pi * w1) ** 2               # Первый член функции.
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)  # Второй член.
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)       # Третий член.
    v = term1 + term2 + term3                     # Сумма всех членов.
    return v                                      # Возвращает значение функции Леви.

def dfLevy(X):
    """Gradient of Levy function for d=2."""
    if not isinstance(X, np.ndarray):             # Приводим к массиву.
        X = np.array(X)
    if X.shape == (2, 1):                         # Преобразуем столбец в вектор.
        X = X.flatten()
    w1 = 1 + (X[0] - 1) / 4.0                     # w1 = 1 + (x1 - 1)/4
    w2 = 1 + (X[1] - 1) / 4.0                     # w2 = 1 + (x2 - 1)/4
    dw1_dx1 = 1/4.0                               # Производная w1 по x1.
    dw2_dx2 = 1/4.0                               # Производная w2 по x2.
    df_dw1 = (                                    # Частная производная f по w1.
        2 * np.sin(np.pi * w1) * np.cos(np.pi * w1) * np.pi +
        2 * (w1 - 1) * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2) +
        (w1 - 1) ** 2 * 10 * 2 * np.sin(np.pi * w1 + 1) * np.cos(np.pi * w1 + 1) * np.pi
    )
    df_dw2 = (                                    # Частная производная f по w2.
        2 * (w2 - 1) * (1 + np.sin(2 * np.pi * w2) ** 2) +
        (w2 - 1) ** 2 * 2 * np.sin(2 * np.pi * w2) * np.cos(2 * np.pi * w2) * 2 * np.pi
    )
    grad_x1 = df_dw1 * dw1_dx1                    # Цепное правило: df/dx1 = df/dw1 * dw1/dx1.
    grad_x2 = df_dw2 * dw2_dx2                    # Цепное правило: df/dx2 = df/dw2 * dw2/dx2.
    return np.array([grad_x1, grad_x2])           # Возвращает градиент как вектор.

def zoom(phi, dphi, alo, ahi, c1, c2):
    j = 1                                         # Счётчик итераций в zoom.
    jmax = 1000                                   # Максимальное число итераций.
    while j < jmax:                               # Цикл уточнения шага.
        a = cinterp(phi, dphi, alo, ahi)          # Новый кандидат шага через кубическую интерполяцию.
        if phi(a) > phi(0) + c1 * a * dphi(0) or phi(a) >= phi(alo):
            ahi = a                               # Сужаем правую границу при нарушении условия Армихо.
        else:
            if abs(dphi(a)) <= -c2 * dphi(0):     # Проверка условия кривизны.
                return a                          # Оба условия Вульфа выполнены — возвращаем шаг.
            if dphi(a) * (ahi - alo) >= 0:        # Если производная положительна — минимум слева.
                ahi = alo                         # Корректируем правую границу.
            alo = a                               # Обновляем левую границу.
        j += 1                                    # Увеличиваем счётчик.
    return a                                      # Аварийный выход.

def cinterp(phi, dphi, a0, a1):
    if np.isnan(dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1))) or (a0 - a1) == 0:
        return a0                                 # Защита от деления на ноль или NaN.
    d1 = dphi(a0) + dphi(a1) - 3 * (phi(a0) - phi(a1)) / (a0 - a1)  # Промежуточный коэффициент.
    if np.isnan(np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))):
        return a0                                 # Защита от некорректного корня.
    d2 = np.sign(a1 - a0) * np.sqrt(d1 ** 2 - dphi(a0) * dphi(a1))  # Второй коэффициент.
    a = a1 - (a1 - a0) * (dphi(a1) + d2 - d1) / (dphi(a1) - dphi(a0) + 2 * d2)  # Формула кубической интерполяции.
    return a                                      # Возвращает интерполированный шаг.

def wolfesearch(f, df, x0, p0, amax, c1, c2):
    x0 = np.atleast_1d(x0).flatten()              # Приводим x0 к одномерному виду.
    p0 = np.atleast_1d(p0).flatten()              # Приводим направление p0 к одномерному виду.
    phi = lambda alpha: f(x0 + alpha * p0)        # Одномерная функция φ(α).
    dphi = lambda alpha: np.dot(p0, df(x0 + alpha * p0))  # Её производная.
    phi0 = phi(0.0)                               # Значение φ(0).
    dphi0 = dphi(0.0)                             # Производная φ'(0).
    a = amax                                      # Начальный шаг — максимальный.
    aprev = 0.0                                   # Предыдущий шаг.
    i = 1                                         # Счётчик итераций.
    imax = 1000                                   # Максимум итераций.
    while i < imax:
        phia = phi(a)
        if (phia > phi0 + c1 * a * dphi0) or ((phia >= phi(aprev)) and (i > 1)):
            a = zoom(phi, dphi, aprev, a, c1, c2) # Запуск zoom при нарушении условий.
            return a
        dphia = dphi(a)
        if abs(dphia) <= -c2 * dphi0:             # Условие кривизны выполнено.
            return a
        if dphia >= 0:                            # Производная положительна — минимум между a и aprev.
            a = zoom(phi, dphi, a, aprev, c1, c2)
            return a
        a = cinterp(phi, dphi, a, amax)           # Интерполяция нового шага.
        aprev = a                                 # Сохраняем предыдущее значение.
        i += 1
    return a                                      # Аварийный выход.

def addnew(arr, el):
    ln = 3                                        # Максимальная длина буфера.
    if len(arr) < ln:                             # Если буфер не полон — добавляем.
        arr.append(el)
    else:                                         # Иначе — сдвигаем влево и заменяем последний.
        for i in range(len(arr)):
            if i < len(arr) - 1:
                arr[i] = arr[i + 1]
            else:
                arr[i] = el

def getz(df, d, y, l, x):
    q = df(x)                                     # Текущий градиент.
    m = []                                        # Список коэффициентов α.
    for i in range(len(l) - 1, -1, -1):           # Обратный проход.
        alpha = l[i] * np.dot(d[i].T, q)          # Вычисляем α_i.
        m.append(alpha)
        q = q - alpha * y[i]                      # Обновляем q.
    H = np.dot(d[-1].T, y[-1]) / np.dot(y[-1].T, y[-1])  # Масштабирование H₀.
    z = H * q                                     # Начальное направление.
    m = m[::-1]                                   # Разворачиваем α.
    for i in range(len(l)):                       # Прямой проход.
        beta = l[i] * np.dot(y[i].T, z)           # Вычисляем β_i.
        z += d[i] * (m[i] - beta)                 # Обновляем z.
    return -z                                     # Возвращаем направление спуска.

def ldfpsearch(f, df, x0, tol):
    if not isinstance(x0, np.ndarray):             # Приводим x0 к массиву.
        x0 = np.array(x0)
    if x0.ndim == 2 and x0.shape[1] == 1:         # Если столбец — делаем вектор.
        x0 = x0.flatten()
    elif x0.ndim > 1:                             # Для общих случаев — сглаживаем.
        x0 = x0.ravel()
    z = -df(x0)                                   # Начальное направление — антиградиент.
    d, y, l = [], [], []                          # Истории векторов d, y и коэффициентов l.
    coordinates = [x0.copy()]                     # Траектория точек.
    xmin = x0.copy()                              # Текущая точка.
    neval = 0                                     # Счётчик вычислений.
    while True:
        g = df(xmin)
        if norm(g) < tol or neval >= 1000:        # Условие остановки.
            break
        xl = xmin.copy()
        alpha = wolfesearch(f, df, xmin, z, 3, tol, 0.1)  # Линейный поиск.
        xmin = xmin + alpha * z                   # Обновление точки.
        coordinates.append(xmin.copy())
        neval += 1
        d_new = xmin - xl                         # Разность позиций.
        y_new = df(xmin) - df(xl)                 # Разность градиентов.
        denom = np.dot(y_new, d_new)              # Скалярное произведение y·d.
        if abs(denom) < 1e-14:                    # Защита от деления на ноль.
            break
        addnew(d, d_new)                          # Обновляем буферы.
        addnew(y, y_new)
        addnew(l, 1.0 / denom)
        z = getz(df, d, y, l, xmin)               # Новое направление.
    fmin = f(xmin)
    return [xmin, fmin, neval, coordinates]       # Возвращаем результат.

def contourPlot(ax, f):
    x1 = np.arange(-4, 4.1, 0.1)                  # Сетка по x.
    m = len(x1)
    y1 = np.arange(-4, 4.1, 0.1)                  # Сетка по y.
    n = len(y1)
    [xx, yy] = np.meshgrid(x1, y1)                # Создаём сетку.
    F = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            X = [xx[i, j], yy[i, j]]              # Координаты точки.
            F[i, j] = f(X)                        # Значение функции.
    nlevels = 20
    ax.contour(xx, yy, F, nlevels, linewidths=1)  # Рисуем контуры.
    ax.set_xlabel('x')                            # Подпись оси X.
    ax.set_ylabel('y')                            # Подпись оси Y.

def dfpDraw(ax, coords, nsteps):
    fSize = 11                                    # Размер шрифта меток.
    x0 = coords[0].flatten()
    ax.text(x0[0] + 0.03, x0[1] + 0.1, str(0), fontsize=fSize)  # Метка старта.
    for i in range(nsteps - 1):
        x0 = coords[i].flatten()
        x1 = coords[i + 1].flatten()
        ax.plot([x0[0], x1[0]], [x0[1], x1[1]], lw=1.2, marker='s', ms=3)  # Отрезок траектории.
    ax.text(x1[0], x1[1] - 0.4, str(nsteps), fontsize=fSize)  # Метка конца.
    ax.scatter(x1[0], x1[1], marker='o', c='red', zorder=12)  # Финальная точка.

def draw(coords, nsteps, f, flag):
    fig, ax = plt.subplots()                      # Создаём график.
    fig.suptitle('Davidon Fletcher Powell method each step visualisation & Countour plot')
    plt.xlim(-4, 4)                               # Пределы по X.
    plt.ylim(-4, 4)                               # Пределы по Y.
    plt.gca().set_aspect('equal', adjustable='box')  # Одинаковый масштаб осей.
    dfpDraw(ax, coords, nsteps)                   # Рисуем траекторию.
    contourPlot(ax, f)                            # Накладываем контуры.
    name = "plot" + flag + ".png"                 # Имя файла.
    fig.savefig(name)                             # Сохраняем.
    ad = "<img width=\"900px\" src=\"/resources/" + name + "\">"  # HTML-тег.
    print(ad)                                     # Выводим тег.

def main():
    print("Sphere function:")                     # Заголовок.
    x0 = np.array([[-1], [-1]])                   # Начальная точка (столбец).
    tol = 1e-5                                    # Точность.
    [xmin, f, neval, coords] = ldfpsearch(fSphere, dfSphere, x0, tol)  # Запуск DFP.
    print(xmin, f, neval)                         # Вывод результата.
    draw(coords, len(coords), fSphere, "s")       # График для сферы.

    print("Levy function:")                       # Заголовок.
    x0 = np.array([[0.5], [0.5]])                 # Новая начальная точка.
    [xmin, f, neval, coords] = ldfpsearch(fLevy, dfLevy, x0, tol)  # Запуск DFP.
    print(xmin, f, neval)                         # Вывод результата.
    draw(coords, len(coords), fLevy, "l")         # График для Леви.

if __name__ == '__main__':
    main()                                        # Запуск программы.