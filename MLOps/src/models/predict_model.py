import torch

def predict_image(model, image, transform, device="cpu"):
    """Предсказание класса изображения."""
    model.eval()
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        _, predicted_class = outputs.max(1)
    return predicted_class.item()
