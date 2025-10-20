
# Слайд 5 — Навчання: спуск градієнтом (2D moons)
# Повна реалізація навчання нейронної мережі з backpropagation
# Демонструємо як мережа навчається на складних даних (місяці)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.common import relu, sigmoid, make_moons

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

# === ПІДГОТОВКА ДАНИХ ===
# Генеруємо складні дані у формі місяців (нелінійно розділені)
X, y = make_moons(n=300, noise=0.18, seed=3)  # 300 точок з шумом
y = y.reshape(-1,1)  # перетворюємо в колонку для сумісності

# === ІНІЦІАЛІЗАЦІЯ МЕРЕЖІ ===
# Використовуємо Xavier/Glorot ініціалізацію для стабільності
rng = np.random.default_rng(1)

# Перший шар: 2 входи -> 8 прихованих нейронів
W1 = rng.normal(0, 0.8, size=(2, 8))  # ваги з нормального розподілу
b1 = np.zeros((1,8))                   # зміщення

# Другий шар: 8 прихованих -> 1 вихід
W2 = rng.normal(0, 0.8, size=(8, 1))  # ваги з нормального розподілу
b2 = np.zeros((1,1))                   # зміщення

def forward(X):
    """
    Forward pass (прохід вперед) через мережу
    
    Параметри:
    - X: вхідні дані [batch_size, input_features]
    
    Повертає:
    - Y: передбачення мережі [batch_size, 1]
    - cache: проміжні значення для backpropagation
    """
    # Шар 1: лінійне перетворення + ReLU активація
    Z1 = X@W1 + b1        # лінійна комбінація: X*W1 + b1
    H1 = np.maximum(0, Z1) # ReLU активація: max(0, Z1)
    
    # Шар 2: лінійне перетворення + Sigmoid активація
    Z2 = H1@W2 + b2       # лінійна комбінація: H1*W2 + b2
    Y = 1/(1+np.exp(-Z2)) # Sigmoid активація: 1/(1+e^(-Z2))
    
    # Зберігаємо проміжні значення для обчислення градієнтів
    cache = (X, Z1, H1, Z2, Y)
    return Y, cache

def loss_fn(y_true, y_prob):
    """
    Binary Cross-Entropy Loss (функція втрат)
    
    L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
    Параметри:
    - y_true: справжні мітки [batch_size, 1]
    - y_prob: передбачені ймовірності [batch_size, 1]
    
    Повертає:
    - L: середня втрата (скаляр)
    """
    eps = 1e-8  # мала константа для уникнення log(0)
    return -np.mean(y_true*np.log(y_prob+eps) + (1-y_true)*np.log(1-y_prob+eps))

def backward(cache, y_true):
    """
    Backward pass (зворотний прохід) - обчислення градієнтів
    
    Використовуємо правило ланцюжка (chain rule) для обчислення:
    ∂L/∂W = ∂L/∂Y * ∂Y/∂Z * ∂Z/∂W
    
    Параметри:
    - cache: проміжні значення з forward pass
    - y_true: справжні мітки
    
    Повертає:
    - grads: градієнти для всіх параметрів (dW1, db1, dW2, db2)
    """
    X, Z1, H1, Z2, Y = cache
    m = X.shape[0]  # розмір батчу
    
    # Градієнти для вихідного шару (Sigmoid + BCE)
    # ∂L/∂Z2 = (Y - y_true) / m
    dZ2 = (Y - y_true)/m
    
    # Градієнти ваг та зміщень вихідного шару
    dW2 = H1.T @ dZ2                    # ∂L/∂W2 = H1^T * dZ2
    db2 = dZ2.sum(axis=0, keepdims=True) # ∂L/∂b2 = sum(dZ2)
    
    # Градієнти для прихованого шару
    dH1 = dZ2 @ W2.T                   # ∂L/∂H1 = dZ2 * W2^T
    
    # Градієнт через ReLU: 0 якщо Z1 <= 0, інакше dH1
    dZ1 = dH1 * (Z1>0)  # ReLU градієнт: 0 для Z1 <= 0
    
    # Градієнти ваг та зміщень прихованого шару
    dW1 = X.T @ dZ1                     # ∂L/∂W1 = X^T * dZ1
    db1 = dZ1.sum(axis=0, keepdims=True) # ∂L/∂b1 = sum(dZ1)
    
    return dW1, db1, dW2, db2

def step(lr, grads):
    """
    Оновлення параметрів за допомогою градієнтного спуску
    
    W_new = W_old - learning_rate * gradient
    
    Параметри:
    - lr: швидкість навчання (learning rate)
    - grads: градієнти (dW1, db1, dW2, db2)
    """
    global W1,b1,W2,b2
    dW1, db1, dW2, db2 = grads
    
    # Оновлюємо параметри: W = W - lr * gradient
    W1 -= lr*dW1; b1 -= lr*db1  # оновлюємо перший шар
    W2 -= lr*dW2; b2 -= lr*db2  # оновлюємо другий шар

# === ЦИКЛ НАВЧАННЯ ===
# Повний цикл навчання з градієнтним спуском
losses = []

print("Початок навчання...")
for epoch in range(2000):
    # 1. Forward pass: обчислюємо передбачення
    yhat, cache = forward(X)
    
    # 2. Обчислюємо втрати
    L = loss_fn(y, yhat)
    losses.append(L)
    
    # 3. Backward pass: обчислюємо градієнти
    grads = backward(cache, y)
    
    # 4. Оновлюємо параметри
    step(0.05, grads)  # learning rate = 0.05
    
    # Виводимо прогрес кожні 400 епох
    if epoch % 400 == 0:
        print(f"Епоха {epoch}: Loss = {L:.4f}")

print(f"Навчання завершено! Фінальна втрата: {losses[-1]:.4f}")

# === ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ===

# Графік 1: Крива навчання (зменшення втрат)
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.title("Training Loss Curve", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(outdir/"slide5_loss.png", bbox_inches="tight", dpi=150)
plt.close()

# Графік 2: Границя прийняття рішень після навчання
# Показуємо як навчена мережа класифікує точки в просторі

# Створюємо сітку точок для побудови границі
xmin, xmax = X[:,0].min()-0.5, X[:,0].max()+0.5
ymin, ymax = X[:,1].min()-0.5, X[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))

# Обчислюємо ймовірності для кожної точки сітки
grid = np.c_[xx.ravel(), yy.ravel()]  # перетворюємо сітку в масив точок
probs, _ = forward(grid)              # отримуємо ймовірності від навченої мережі
ZZ = probs.reshape(xx.shape)         # перетворюємо назад у форму сітки

# Візуалізуємо результат
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, ZZ, levels=20, alpha=0.6, cmap='RdYlBu')
plt.scatter(X[:,0], X[:,1], c=y.ravel(), cmap='RdYlBu', edgecolors='black', linewidth=0.5)
plt.title("MLP Decision Boundary After Training", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.colorbar(label="Predicted Probability")
plt.savefig(outdir/"slide5_boundary_trained.png", bbox_inches="tight", dpi=150)
plt.close()

print("Saved: outputs/slide5_loss.png, outputs/slide5_boundary_trained.png")
