import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

# Путь к модели
MODEL_PATH = "models/mobilenetv2_100.onnx"

# Загрузка модели
session = ort.InferenceSession(MODEL_PATH)

# Препроцессинг
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).numpy()
    return image

# Тестовое изображение
image_path = "path_to_test_image.jpg"  # Укажите путь к изображению
input_data = preprocess_image(image_path)

# Инференс
outputs = session.run(None, {"input": input_data})
print("Результат:", outputs)
