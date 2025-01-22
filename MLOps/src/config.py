import os

# Пути к папкам
links_folder = "./data/animal_links"
dataset_folder = "./data/animal_dataset"

# Список животных
animals_list = ['lion', 'zebra', 'rabbit', 'bear', 'cow', 'wolf', 'cat', 'dog', 'horse', 'deer']

# Создание папок
os.makedirs(links_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)
