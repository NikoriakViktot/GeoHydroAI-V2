import pandas as pd

# 1. Завантажуємо Parquet
df = pd.read_parquet("/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_3857_no_geometry.parquet")

# 2. Замінюємо всі кириличні “е” на латиничні “e” в усіх назвах колонок
df.columns = [col.replace("е", "e") for col in df.columns]

# 3. Зберігаємо новий файл
df.to_parquet("/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_3857_1.parquet")