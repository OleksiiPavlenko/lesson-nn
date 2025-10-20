
# Слайд 4 — MLP: forward-pass без бібліотек (2 шари)
# Демонстрація багатошарового перцептрона (MLP) з 2 шарами
# Показуємо як дані проходять через мережу та як виглядають активації

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')
from utils.common import relu, make_blob

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

# === ПІДГОТОВКА ДАНИХ ===
# Генеруємо синтетичні дані: 2 кластери точок
# Один кластер навколо (0,0), другий навколо (2,2)
X, y = make_blob(n=200, centers=((0,0),(2,2)), seed=1)

# === ІНІЦІАЛІЗАЦІЯ МЕРЕЖІ ===
# Створюємо випадкові ваги для демонстрації (без навчання)
rng = np.random.default_rng(0)

# Перший шар: 2 входи -> 8 прихованих нейронів
W1 = rng.normal(0, 1, size=(2, 8))  # ваги: [2 входи] x [8 нейронів]
b1 = np.zeros(8)                     # зміщення (bias) для кожного нейрона

# Другий шар: 8 прихованих -> 1 вихід
W2 = rng.normal(0, 1, size=(8, 1))   # ваги: [8 нейронів] x [1 вихід]
b2 = np.zeros(1)                     # зміщення для виходу

# === FORWARD PASS (ПРОХІД ВПЕРЕД) ===
# Крок 1: Прихований шар
# H = ReLU(X * W1 + b1)
# X: [200, 2] - вхідні дані
# W1: [2, 8] - ваги першого шару
# b1: [8] - зміщення
# H: [200, 8] - активації прихованого шару
H = relu(X @ W1 + b1)     # прихований шар з ReLU активацією

# Крок 2: Вихідний шар
# logits = H * W2 + b2
# H: [200, 8] - активації прихованого шару
# W2: [8, 1] - ваги другого шару
# b2: [1] - зміщення виходу
# logits: [200, 1] - "сирі" оцінки
logits = H @ W2 + b2      # вихід без активації

# Крок 3: Сигмоїдна активація для отримання ймовірностей
# probs = sigmoid(logits) = 1 / (1 + exp(-logits))
probs = 1/(1+np.exp(-logits))

# === ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ===

# Графік 1: Активації прихованого шару
# Показуємо як вхідні дані трансформуються в прихованому шарі
# Використовуємо тільки перші 2 нейрони для 2D візуалізації
plt.figure()
plt.scatter(H[:,0], H[:,1], c=y)  # H[:,0], H[:,1] - активації перших 2 нейронів
plt.title("Hidden activations (neurons 1&2)")
plt.xlabel("Neuron 1 activation")
plt.ylabel("Neuron 2 activation")
plt.savefig(outdir/"slide4_hidden.png", bbox_inches="tight")
plt.close()

# Графік 2: Границя прийняття рішень
# Показуємо як мережа класифікує точки в просторі
# (випадкові ваги = погана класифікація, але показує принцип)

# Створюємо сітку точок для побудови границі
xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))

# Обчислюємо активації для кожної точки сітки
# np.c_[xx.ravel(), yy.ravel()] - перетворюємо сітку в масив точок
HH = relu(np.c_[xx.ravel(), yy.ravel()] @ W1 + b1)  # активації прихованого шару
ZZ = 1/(1+np.exp(-(HH @ W2 + b2)))                  # ймовірності для кожної точки

# Візуалізуємо результат
plt.figure()
plt.contourf(xx, yy, ZZ.reshape(xx.shape), alpha=0.3)  # зафарбовуємо області
plt.scatter(X[:,0], X[:,1], c=y)                       # показуємо вхідні дані
plt.title("Random MLP (no training)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(outdir/"slide4_boundary.png", bbox_inches="tight")
plt.close()

print("Saved: outputs/slide4_hidden.png, outputs/slide4_boundary.png")
