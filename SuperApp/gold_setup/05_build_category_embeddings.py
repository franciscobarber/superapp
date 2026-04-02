# Databricks Notebook — Gold Setup Step 5
# Build category centroids from labeled products.
# Run this ONCE after notebook 02, then re-run whenever you approve new items
# from the review queue (notebook 07) to keep centroids up to date.
#
# Output: workspace.superapp.gold_category_centroids
#   id_categoria | nombre | n_productos | centroid_json (normalized float list)

# COMMAND ----------

%pip install sentence-transformers --quiet
dbutils.library.restartPython()

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
from pyspark.sql.functions import col

print("Libraries loaded.")

# COMMAND ----------

# Load all classified products (any method) with their names from silver.
# We join silver_prices to get the human-readable product name.
labeled = spark.sql("""
    SELECT DISTINCT
        pc.id_producto,
        pc.id_categoria,
        gc.nombre      AS categoria_nombre,
        sp.producto
    FROM workspace.superapp.gold_productos_categorias pc
    JOIN workspace.superapp.gold_categorias gc
        ON pc.id_categoria = gc.id_categoria
    JOIN workspace.superapp.silver_prices sp
        ON pc.id_producto = sp.id_producto
    WHERE pc.id_categoria IS NOT NULL
      AND gc.nombre != 'sin_clasificar'
      AND sp.producto IS NOT NULL
""").toPandas()

print(f"Labeled products loaded:  {len(labeled):,}")
print(f"Categories represented:   {labeled['categoria_nombre'].nunique()}")
print()
print(labeled.groupby('categoria_nombre').size().sort_values(ascending=False).to_string())

# COMMAND ----------

# Embed all labeled product names.
# paraphrase-multilingual-MiniLM-L12-v2 is 50 MB, multilingual, fast on CPU.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print(f"Generating embeddings for {len(labeled):,} products...")
embeddings = model.encode(
    labeled['producto'].tolist(),
    show_progress_bar=True,
    batch_size=128
)
labeled['embedding'] = list(embeddings)
print(f"Done. Embedding dim: {embeddings.shape[1]}")

# COMMAND ----------

# Compute per-category centroid = mean of member embeddings, L2-normalized.
# Normalization makes cosine similarity = dot product (faster later).
print("Computing centroids...")
centroids = []

for cat_nombre, group in labeled.groupby('categoria_nombre'):
    vecs = np.array(group['embedding'].tolist())          # (n, dim)
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm

    centroids.append({
        'id_categoria': int(group['id_categoria'].iloc[0]),
        'nombre':       cat_nombre,
        'n_productos':  int(len(group)),
        'centroid_json': json.dumps(centroid.tolist()),
    })

centroids_pd = pd.DataFrame(centroids)
print(f"Centroids computed: {len(centroids_pd)}")
print()
print(centroids_pd[['nombre', 'n_productos']].sort_values('n_productos', ascending=False).to_string())

# COMMAND ----------

# Persist centroids to Delta — overwrite so each run replaces the previous snapshot.
(spark.createDataFrame(centroids_pd[['id_categoria', 'nombre', 'n_productos', 'centroid_json']])
     .write.mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable("workspace.superapp.gold_category_centroids"))

print(f"Saved {len(centroids_pd)} centroids → workspace.superapp.gold_category_centroids")
