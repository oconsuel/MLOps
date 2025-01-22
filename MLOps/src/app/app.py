import gradio as gr
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np

# Путь к вашей ONNX-модели
MODEL_PATH = "notebooks/models/mobilenetv2_100.onnx "
CLASS_NAMES = ['lion', 'zebra', 'rabbit', 'bear', 'cow', 'wolf', 'cat', 'dog', 'horse', 'deer']  # Замените на список ваших классов

# Загрузка модели ONNX
session = ort.InferenceSession(MODEL_PATH)

# Подготовка изображения
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0).numpy()
    return image

# Функция инференса
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    input_tensor = preprocess_image(image)
    outputs = session.run(None, {"input": input_tensor})
    predicted_class = np.argmax(outputs[0], axis=1)[0]
    return CLASS_NAMES[predicted_class]

# Интерфейс Gradio
def main():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="numpy", label="Загрузите изображение"),
        outputs=gr.Label(label="Класс изображения"),
        title="Классификация изображений",
        description="Это приложение использует модель MobileNet, экспортированную в ONNX, для классификации изображений."
    )
    interface.launch()

if __name__ == "__main__":
    main()
