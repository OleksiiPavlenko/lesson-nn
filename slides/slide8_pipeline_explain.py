
# Слайд 8 — Пояснення пайплайну «зображення → модель → клас»
# Візуальна схема процесу класифікації зображень
# Показуємо кожен крок від сирого зображення до фінального результату

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

# === СТВОРЕННЯ СХЕМИ ПАЙПЛАЙНУ ===
# Створюємо зображення з темним фоном для схеми
img = Image.new("RGB", (1200, 400), (34,45,80))  # темно-синій фон
d = ImageDraw.Draw(img)

def box(x, y, w, h, text):
    """
    Малюємо прямокутник з текстом для схеми пайплайну
    
    Параметри:
    - x, y: координати лівого верхнього кута
    - w, h: ширина та висота прямокутника
    - text: текст для відображення
    """
    # Малюємо рамку прямокутника
    d.rectangle([x, y, x+w, y+h], outline=(220,230,250), width=4)
    # Додаємо текст всередину
    d.text((x+10, y+10), text, fill=(230,240,255))

# === КРОКИ ПАЙПЛАЙНУ КЛАСИФІКАЦІЇ ===

# Крок 1: Вхідне зображення
# Сирі пікселі зображення у форматі H×W×C (висота×ширина×канали)
box(20, 120, 250, 160, "Input image\n(pixels HxWxC)")

# Крок 2: Попередня обробка (Preprocessing)
# - Зміна розміру до стандартного (224×224)
# - Нормалізація значень пікселів (0-255 → 0-1)
# - Стандартизація (ImageNet mean/std)
box(320, 100, 280, 200, "Preprocess\n(resize, normalize)")

# Крок 3: CNN (Convolutional Neural Network)
# ResNet18 архітектура:
# - Convolutional layers (виявлення локальних ознак)
# - Pooling layers (зменшення розмірності)
# - Residual connections (глибокі мережі)
# - Batch normalization (стабілізація)
box(640, 80, 260, 240, "CNN (ResNet18)\n(convolutions, pooling)")

# Крок 4: Вихідний шар
# - Fully Connected layer (фінальна класифікація)
# - Softmax (ймовірності класів)
# - Top-k результати (найкращі передбачення)
box(940, 120, 240, 160, "Output layer\n(top-k classes)")

# === ДОДАЄМО ЗАГОЛОВОК ===
# Загальний опис пайплайну
d.text((30, 40), "Pipeline: image → preprocess → CNN → label", fill=(240,250,255))

# Зберігаємо схему
img.save(outdir/"slide8_pipeline.png")
print("Saved: outputs/slide8_pipeline.png")
