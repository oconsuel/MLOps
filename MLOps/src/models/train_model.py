import torch
from torch import nn
from torchvision import models

def train_model(train_loader, test_loader, num_epochs=10, device="cpu"):
    """Обучение модели MobileNetV2."""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_loader.dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return model

def export_to_onnx(model, output_path="model.onnx", input_size=(1, 3, 224, 224)):
    """Экспорт модели в формат ONNX."""
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(model, dummy_input, output_path, verbose=True)
    print(f"Модель экспортирована в {output_path}")
