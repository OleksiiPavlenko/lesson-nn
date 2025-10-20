
# Слайд 5 (дод.) — Активації: sigmoid, tanh, ReLU
# Демонстрація основних функцій активації в нейронних мережах
# Кожна функція має свої особливості та застосування

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.common import sigmoid, tanh, relu

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

# === ПІДГОТОВКА ДАНИХ ===
# Створюємо діапазон значень для побудови графіків
x = np.linspace(-6, 6, 400)  # від -6 до 6 з 400 точками

# === ВІЗУАЛІЗАЦІЯ ФУНКЦІЙ АКТИВАЦІЇ ===
plt.figure()

# 1. SIGMOID: σ(x) = 1 / (1 + e^(-x))
# - Діапазон: (0, 1)
# - S-подібна крива
# - Використовується в останніх шарах для ймовірностей
# - Проблема: "vanishing gradient" при великих |x|
plt.plot(x, sigmoid(x), label="sigmoid", linewidth=2)

# 2. TANH: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# - Діапазон: (-1, 1)
# - Центрована версія sigmoid
# - Краще для прихованих шарів
# - Також має проблему "vanishing gradient"
plt.plot(x, tanh(x), label="tanh", linewidth=2)

# 3. ReLU: ReLU(x) = max(0, x)
# - Діапазон: [0, +∞)
# - Проста та ефективна
# - Вирішує проблему "vanishing gradient"
# - Найпопулярніша в сучасних мережах
# - Проблема: "dying ReLU" (нульові градієнти для x < 0)
plt.plot(x, relu(x), label="ReLU", linewidth=2)

# Налаштування графіка
plt.title("Activation functions", fontsize=14)
plt.xlabel("Input (x)", fontsize=12)
plt.ylabel("Output", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(-6, 6)
plt.ylim(-1.2, 1.2)

# Зберігаємо графік
plt.savefig(outdir/"slide5_activations.png", bbox_inches="tight", dpi=150)
plt.close()

print("Saved: outputs/slide5_activations.png")
