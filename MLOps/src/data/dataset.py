import os
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from duckduckgo_search import DDGS
from src.config import links_folder, dataset_folder

def generate_links(query_list, max_results=50):
    for query in query_list:
        links_file = os.path.join(links_folder, f"{query}.txt")
        print(f"Генерация ссылок для {query}, максимум: {max_results}")
        
        with DDGS() as ddgs:
            results = ddgs.images(
                f"{query} photo", region="wt-wt", size="Medium", max_results=max_results
            )
            unique_links = set()
            
            with open(links_file, "w", encoding="utf-8") as f:
                for result in results:
                    link = result.get("image")
                    if link and link not in unique_links:
                        unique_links.add(link)
                        f.write(f"{link}\n")
        print(f"Ссылки для {query} сохранены в {links_file}")

def download_images(animals, max_images=30):
    for animal in animals:
        links_file = os.path.join(links_folder, f"{animal}.txt")
        output_folder = os.path.join(dataset_folder, animal)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(links_file):
            print(f"Файл со ссылками для {animal} не найден. Пропускаем...")
            continue

        with open(links_file, "r", encoding="utf-8") as f:
            links = f.readlines()

        count = 0
        for i, link in enumerate(links):
            if count >= max_images:
                break
            link = link.strip()

            try:
                response = requests.get(link, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img_path = os.path.join(output_folder, f"{animal}_{i+1}.jpg")
                img.save(img_path, format="JPEG")
                count += 1
                print(f"Скачано: {img_path}")

            except (requests.RequestException, UnidentifiedImageError) as e:
                print(f"Ошибка при скачивании {link}: {e}")
