from PIL import Image
from torchvision import transforms as T

# Трансформации для тренировочного набора
train_transforms = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Трансформации для тестового набора
test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path, transform=test_transforms):
    """Применение трансформаций к изображению."""
    image = Image.open(image_path).convert("RGB")
    return transform(image)
