import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def F1(x):
    return -2 * np.sin(np.sqrt(abs(x / 2 + 10))) - x * np.sin(np.sqrt(abs(x - 10)))

def dF1(x):
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

def F(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def dF(x):
    """
    Аналитическая производная f2(x) = x^2 - 10*cos(0.5*pi*x) - 110.
    f2'(x) = 2x + 5*pi*sin(0.5*pi*x)
    """
    return 2 * x + 5 * np.pi * np.sin(0.5 * np.pi * x)



def armijo_1d(f, df, x, p, a=1.0, c1=0.1, b=0.5, max_iter=100):
    """Line search по условию Армиджо для 1D."""
    phi0 = f(x)
    dphi0 = df(x) * p  # скалярное произведение в 1D

    k = 0
    while k < max_iter:
        x_new = x + a * p
        if f(x_new) <= phi0 + c1 * a * dphi0:
            return a
        a *= b
        k += 1
    return a  # возвращаем последнее значение даже при неудаче

def prsearch_1d(f, df, x0, tol=1e-6, max_iter=1000):
    """
    Метод сопряжённых градиентов (Полак–Рибьер) для 1D.
    В 1D вырождается в градиентный спуск с адаптивным шагом.
    """
    x = x0
    coords = [x]
    g_old = df(x)
    neval = 0

    while neval < max_iter:
        p = -g_old  # направление антиградиента

        # Подбор шага
        alpha = armijo_1d(f, df, x, p, a=1.0, c1=0.1, b=0.5)

        # Обновление точки
        x_new = x + alpha * p
        g_new = df(x_new)

        coords.append(x_new)

        # Проверка сходимости
        if abs(g_new) < tol:
            break

        # Polak-Ribiere beta (в 1D — формально)
        beta_pr = (g_new * (g_new - g_old)) / (g_old ** 2 + 1e-15)
        # В 1D новое направление всё равно будет -g_new, но оставим для соответствия
        p = -g_new + beta_pr * p

        # Обновляем состояние
        x = x_new
        g_old = g_new
        neval += 1

    fmin = f(x)
    return x, fmin, neval + 1, coords

def draw_1d(coords, f, flag):
    """Визуализация траектории оптимизации."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.linspace(-2, 10, 2000)
    y_vals = [f(x) for x in x_vals]
    ax.plot(x_vals, y_vals, 'b-', linewidth=1.5, label='F(x)')

    # Отметим все точки траектории
    for i, x in enumerate(coords):
        ax.plot(x, f(x), 'ro', markersize=4)
        if i % max(1, len(coords)//10) == 0 or i == len(coords)-1:  # не перегружать подписями
            ax.text(x, f(x) + 0.8, str(i), fontsize=8, ha='center')

    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title('Optimization path (1D Polak-Ribiere with Armijo)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    name = f"plot_{flag}.png"
    fig.savefig(name, dpi=150, bbox_inches='tight')
    print(f'<img width="900px" src="/resources/{name}">')

def main():
    print("Optimizing 1D function F(x) using 1D Polak-Ribiere method with Armijo line search")
    x0 = 5.0  # начальная точка (скаляр!)
    tol = 1e-6

    xmin, fmin, neval, coords = prsearch_1d(F, dF, x0, tol=tol)

    print(f"\nResult:")
    print(f"  xmin = {xmin:.8f}")
    print(f"  fmin = {fmin:.8f}")
    print(f"  iterations = {neval}")
    print(f"  final |f'(xmin)| = {abs(dF(xmin)):.2e}")

    draw_1d(coords, F, "1d")

if __name__ == '__main__':
    main()