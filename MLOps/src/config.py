import os

# Пути к папкам
links_folder = "./data/animal_links"
dataset_folder = "./data/animal_dataset_split"

# Список животных
animals_list = ['lion', 'zebra', 'rabbit', 'bear', 'cow', 'wolf', 'cat', 'dog', 'horse', 'deer']

# Создание папок
os.makedirs(links_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)

MODEL_PATH = "./models/"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
