
# Слайд 6 — CNN інференс: ResNet18 (якщо доступний) або легкий fallback
# Демонстрація інференсу зглибокої нейронної мережі на реальному зображенні
# Показуємо як CNN класифікує зображення та виводить ймовірності

import argparse, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Створюємо папку для збереження результатів
outdir = Path("outputs"); outdir.mkdir(exist_ok=True, parents=True)

def classify_fallback(img: Image.Image):
    """
    Простий fallback класифікатор (якщо PyTorch недоступний)
    
    Принцип роботи: аналізуємо домінуючий колір у зображенні
    - Обчислюємо середні значення для кожного каналу (R, G, B)
    - Класифікуємо за домінуючим каналом
    
    Параметри:
    - img: PIL Image об'єкт
    
    Повертає:
    - label: назва класу ("red-ish", "green-ish", "blue-ish")
    - score: впевненість (0-1)
    """
    # Змінюємо розмір до стандартного для CNN (224x224)
    arr = np.asarray(img.resize((224,224))).astype(np.float32)/255.0
    
    # Обчислюємо середні значення для кожного каналу
    m = arr.mean(axis=(0,1))  # [R_mean, G_mean, B_mean]
    
    # Знаходимо індекс каналу з найбільшим середнім значенням
    idx = int(np.argmax(m))
    
    # Визначаємо клас за домінуючим каналом
    names = ["red-ish", "green-ish", "blue-ish"]
    return names[idx], float(m[idx])

def try_torch(img: Image.Image):
    try:
        # Імпортуємо PyTorch та необхідні модулі
        import torch, torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights
        
        # Завантажуємо попередньо навчену модель ResNet18
        weights = ResNet18_Weights.DEFAULT  # ImageNet weights
        model = resnet18(weights=weights).eval()  # переводимо в режим інференсу
        
        # Отримуємо стандартні трансформації для ImageNet
        preprocess = weights.transforms()
        
        # Підготовка зображення для моделі
        x = preprocess(img).unsqueeze(0)  # додаємо batch dimension
        
        # Інференс без обчислення градієнтів (швидше)
        with torch.no_grad():
            logits = model(x)                    # отримуємо "сирі" оцінки
            prob = torch.softmax(logits, dim=1)[0]  # перетворюємо в ймовірності
            topk = torch.topk(prob, 5)          # отримуємо топ-5 результатів
            
            # Отримуємо назви класів та їх ймовірності
            classes = [weights.meta["categories"][int(i)] for i in topk.indices]
            scores = [float(p) for p in topk.values]
        
        return list(zip(classes, scores))
    except Exception as e:
        # Якщо PyTorch недоступний, повертаємо None
        return None

def main(image_path):
    """
    Основна функція для класифікації зображення
    
    Параметри:
    - image_path: шлях до зображення
    """
    # Завантажуємо та підготовляємо зображення
    img = Image.open(image_path).convert("RGB")
    
    # Спроба використати ResNet18
    res = try_torch(img)
    
    if res is None:
        # Fallback: простий класифікатор за кольором
        label, score = classify_fallback(img)
        print(f"Fallback label: {label} (score {score:.3f})")
        
        # Зберігаємо результат з fallback класифікатором
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Fallback Classification: {label} ({score:.2f})", fontsize=12)
        plt.savefig(outdir/"slide6_fallback.png", bbox_inches="tight", dpi=150)
        plt.close()
    else:
        # Успішна класифікація з ResNet18
        print("Top-5 predictions:")
        for c, s in res:
            print(f"{c:>25s}  {s:.3f}")
        
        # Зберігаємо результат з ResNet18
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        title = ", ".join([f"{c}:{s:.2f}" for c, s in res[:3]])
        plt.title(f"ResNet18: {title}", fontsize=12)
        plt.savefig(outdir/"slide6_resnet18.png", bbox_inches="tight", dpi=150)
        plt.close()

if __name__ == "__main__":
    # Налаштування аргументів командного рядка
    ap = argparse.ArgumentParser(description="CNN Image Classification Demo")
    ap.add_argument("--image", type=str, default="data/sample.jpg", 
                   help="Path to input image")
    args = ap.parse_args()
    main(args.image)
