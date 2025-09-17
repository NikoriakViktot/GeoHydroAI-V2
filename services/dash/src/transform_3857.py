import pyarrow.parquet as pq
import pyarrow as pa
from pyproj import Transformer



parquet_path = "/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_4326.parquet"
output_path = "/mnt/c/Users/5302/PycharmProjects/GeoHydroAI/data/tracks_3857.parquet"



pq_file = pq.ParquetFile(output_path)
print("📦 Колонки у файлі:")
print(pq_file.schema.names)
batch = next(pq_file.iter_batches(batch_size=5))
table = batch.to_pandas()

# Показати тільки координати
print(table[["x", "y", "x_merc", "y_merc"]])
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from shapely import wkb
from pyproj import Transformer

def transform_parquet_geometry(parquet_path, output_path, batch_size=100_000):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    src = pq.ParquetFile(parquet_path)
    writer = None

    for batch in src.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()

        # Витяг координат з WKB → x, y
        geometries = df["geometry"].apply(lambda g: wkb.loads(g) if g else None)
        df["x"] = geometries.apply(lambda geom: geom.x if geom else None)
        df["y"] = geometries.apply(lambda geom: geom.y if geom else None)

        # Трансформація до EPSG:3857
        x_merc, y_merc = transformer.transform(df["x"].values, df["y"].values)
        df["x_merc"] = x_merc
        df["y_merc"] = y_merc

        # Зберігаємо як parquet
        table_out = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table_out.schema)
        writer.write_table(table_out)

    if writer:
        writer.close()
        print(f"✅ Трансформовано та збережено: {output_path}")



# transform_parquet_geometry(parquet_path, output_path)
