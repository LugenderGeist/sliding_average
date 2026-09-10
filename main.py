import numpy as np
import matplotlib.pyplot as plt

# Параметры
L = 10.0          # интервал определения функции
N = 100           # число точек дискретизации
sigma = 0.5       # уровень шума

# Дискретная сетка
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Аналитическая функция
def f(x):
    return np.sin(2 * np.pi * x / L)
y_true = f(x)

# Равномерно распределённый шум в диапазоне [-sigma, sigma]
np.random.seed(42)  # для воспроизводимости
noise = np.random.uniform(-sigma, sigma, size=N)
y_noisy = y_true + noise

# Функция скользящего среднего
def moving_average(y, window):
    n = len(y)
    half = window // 2
    y_smooth = np.zeros(n)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        y_smooth[i] = np.mean(y[left:right])
    return y_smooth

# Сглаживание по трем точкам
y_sa3 = moving_average(y_noisy, 3)
y_sa3_twice = moving_average(y_sa3, 3)   # повторное сглаживание

# Сглаживание по пяти точкам
y_sa5 = moving_average(y_noisy, 5)
y_sa5_twice = moving_average(y_sa5, 5)

# Метрики: RMSE и MAE
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b):
    return np.mean(np.abs(a - b))

signals = {
    'Зашумлённый'   : y_noisy,
    '3 точки'       : y_sa3,
    '3 точки×2'     : y_sa3_twice,
    '5 точек'       : y_sa5,
    '5 точек×2'     : y_sa5_twice,
}

print(f"{'Сигнал':<15}{'RMSE':>10}{'MAE':>10}")
print("-" * 35)
for name, y in signals.items():
    print(f"{name:<15}{rmse(y, y_true):>10.4f}{mae(y, y_true):>10.4f}")

# ГРАФИКИ
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['mathtext.fontset'] = 'stix'

# Исходная и зашумленная функции
plt.figure(figsize=(10, 5))
plt.plot(x, y_true,  'k-', lw=2, label='Исходная f(x)')
plt.plot(x, y_noisy, 'k-', lw=1, label='Зашумленная f(x)')
plt.title('Исходная и зашумленная функции')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# Шум
plt.figure(figsize=(10, 4))
plt.plot(x, noise, 'k-', lw=1)
plt.axhline(0, color='k', lw=0.5)
plt.title(f'Равномерный шум, разброс ±{sigma}')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True)
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# Зашумленная и сглаженная (3 точки)
plt.figure(figsize=(10, 5))
plt.plot(x, y_true, 'k--', lw=1.5, label='Аналитическая')
plt.plot(x, y_noisy, 'k-', lw=1, alpha=0.8, label='Зашумленная')
plt.plot(x, y_sa3,   'k-', lw=2, label='Сглаженная')
plt.title('Сглаживание скользящим средним по 3 точкам')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# Сравнение после повторного сглаживания
plt.figure(figsize=(10, 5))
plt.plot(x, y_true,       'k--', lw=1.5, label='Аналитическая')
plt.plot(x, y_sa3,        'k-',  lw=1, label='Сглаженная')
plt.plot(x, y_sa3_twice,  'k-',  lw=2, label='Сглаженная дважды')
plt.title('Повторное сглаживание по 3 точкам')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# Зашумленная и сглаженная (5 точек)
plt.figure(figsize=(10, 5))
plt.plot(x, y_true,       'k--', lw=1.5, label='Аналитическая')
plt.plot(x, y_noisy, 'k-', lw=1, alpha=0.8, label='Зашумленная')
plt.plot(x, y_sa5,   'k-', lw=2, label='Сглаженная')
plt.title('Сглаживание скользящим средним по 5 точкам')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# Сравнение после повторного сглаживания
plt.figure(figsize=(10, 5))
plt.plot(x, y_true,       'k--', lw=1.5, label='Аналитическая')
plt.plot(x, y_sa5,        'k-',  lw=1, label='Сглаженная')
plt.plot(x, y_sa5_twice,  'k-',  lw=2, label='Сглаженная дважды')
plt.title('Повторное сглаживание по 3 точкам')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.grid(True); plt.legend()
plt.xlim(0, 10); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()