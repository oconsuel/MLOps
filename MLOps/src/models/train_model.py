import os
import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.features.build_features import get_transforms
from src.config import dataset_folder, MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

def export_model_to_onnx(model, model_name, export_path):
    # Создаем путь для сохранения
    os.makedirs(export_path, exist_ok=True)

    # Путь для сохранения ONNX-модели
    onnx_path = os.path.join(export_path, f"{model_name}.onnx")
    
    # Создаем фиктивный вход (размер изображения 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Экспортируем модель в ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Модель успешно экспортирована в ONNX: {onnx_path}")


def train_model(model_name='mobilenetv2_100'):
    # Подготовка данных
    train_dataset = ImageFolder(os.path.join(dataset_folder, 'train'), transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = ImageFolder(os.path.join(dataset_folder, 'test'), transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Загрузка модели MobileNet
    model = timm.create_model(model_name, pretrained=True, num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучение модели
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader)}")

    # Тестирование модели
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Точность модели на тестовом наборе: {100 * correct / total}%")

    # Сохранение модели
    os.makedirs(MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'{model_name}.pth'))
    print(f"Модель {model_name} сохранена.")

    # Экспорт модели в ONNX
    export_model_to_onnx(model, model_name, MODEL_PATH)