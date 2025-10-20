

# Слайд 3 — Перцептрон на простому прикладі (логічні функції)
# Демонстрація найпростішої нейронної мережі - перцептрона
# Показуємо як перцептрон може навчитися логічним операціям AND та OR

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

def perceptron_train(X, y, lr=0.1, epochs=25):
    """
    Навчання перцептрона за допомогою алгоритму перцептрона
    
    Параметри:
    - X: вхідні дані (n_samples, n_features)
    - y: цільові значення (0 або 1)
    - lr: швидкість навчання (learning rate)
    - epochs: кількість епох навчання
    
    Повертає:
    - w: навчені ваги [bias, weight1, weight2, ...]
    """
    # Ініціалізуємо ваги нулями (+1 для bias)
    w = np.zeros(X.shape[1]+1)  # +bias
    
    # Алгоритм перцептрона: повторюємо навчання epochs разів
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            # Обчислюємо зважену суму: z = w1*x1 + w2*x2 + bias
            z = np.dot(w[1:], xi) + w[0]
            
            # Функція активації (крокова): 1 якщо z >= 0, інакше 0
            pred = 1 if z >= 0 else 0
            
            # Оновлюємо ваги за правилом перцептрона:
            # w_new = w_old + learning_rate * (target - prediction) * input
            w[1:] += lr*(yi - pred)*xi  # оновлюємо ваги входів
            w[0]  += lr*(yi - pred)      # оновлюємо bias
    return w

def plot_decision(X, y, w, title, fname):
    """
    Візуалізація границі прийняття рішень перцептрона
    
    Параметри:
    - X: вхідні дані для візуалізації
    - y: цільові значення (кольори точок)
    - w: навчені ваги [bias, weight1, weight2]
    - title: заголовок графіка
    - fname: ім'я файлу для збереження
    """
    # Визначаємо межі для візуалізації
    xmin, xmax = X[:,0].min()-0.5, X[:,0].max()+0.5
    ymin, ymax = X[:,1].min()-0.5, X[:,1].max()+0.5
    
    # Створюємо сітку точок для побудови границі
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    
    # Обчислюємо значення для кожної точки сітки: z = bias + w1*x + w2*y
    Z = w[0] + w[1]*xx + w[2]*yy
    
    # Створюємо графік
    plt.figure()
    # Зафарбовуємо області де z >= 0 (клас 1) та z < 0 (клас 0)
    plt.contourf(xx, yy, (Z>=0).astype(float), alpha=0.3)
    # Показуємо вхідні дані різними кольорами
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title(title)
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.savefig(outdir/fname, bbox_inches="tight")
    plt.close()

# === ДЕМОНСТРАЦІЯ ЛОГІЧНИХ ВОРОТ ===

# AND gate: повертає 1 тільки якщо обидва входи = 1
# Таблиця істинності: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])  # всі можливі комбінації входів
y_and = np.array([0,0,0,1])                   # відповідні виходи для AND
w_and = perceptron_train(X_and, y_and)        # навчаємо перцептрон
plot_decision(X_and, y_and, w_and, "Perceptron: AND gate", "slide3_and.png")

# OR gate: повертає 1 якщо хоча б один вхід = 1
# Таблиця істинності: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
y_or = np.array([0,1,1,1])                    # відповідні виходи для OR
w_or = perceptron_train(X_and, y_or)          # навчаємо на тих же входах
plot_decision(X_and, y_or, w_or, "Perceptron: OR gate", "slide3_or.png")

print("Saved: outputs/slide3_and.png, outputs/slide3_or.png")
