# Databricks Notebook — Gold Setup Step 6
# Classify unclassified products using category centroid similarity.
#
# Confidence routing:
#   similarity >= 0.85  →  auto-assign (metodo='embedding_auto')
#   0.60 – 0.85         →  LLM confirms or redirects → review queue
#   < 0.60              →  LLM proposes new category name → review queue
#
# Prerequisites: run 05_build_category_embeddings.py first.
# Outputs:
#   workspace.superapp.gold_productos_categorias  (auto-assigned rows appended)
#   workspace.superapp.gold_review_queue          (uncertain rows for notebook 07)

# COMMAND ----------

%pip install sentence-transformers --quiet
dbutils.library.restartPython()

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime
from pyspark.sql.functions import col

# ── Thresholds ────────────────────────────────────────────────────────────────
CONFIDENCE_AUTO = 0.85   # above: auto-assign without LLM
CONFIDENCE_LLM  = 0.60   # above: LLM confirms; below: LLM proposes new category

# ── LLM endpoint (working in your workspace) ─────────────────────────────────
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

BATCH_SIZE = 500   # products per embedding batch

print("Config loaded.")
print(f"  Auto-assign threshold: {CONFIDENCE_AUTO}")
print(f"  LLM confirmation threshold: {CONFIDENCE_LLM}")

# COMMAND ----------

# Load centroids built by notebook 05.
centroids_pd = spark.table("workspace.superapp.gold_category_centroids").toPandas()
centroids_pd['centroid'] = centroids_pd['centroid_json'].apply(json.loads).apply(np.array)

centroid_matrix = np.array(centroids_pd['centroid'].tolist())   # (n_cats, dim)
category_names  = centroids_pd['nombre'].tolist()
cat_id_map      = dict(zip(centroids_pd['nombre'], centroids_pd['id_categoria']))

print(f"Centroids loaded: {len(centroids_pd)} categories")
print(f"Centroid matrix shape: {centroid_matrix.shape}")

# COMMAND ----------

# Products not yet in gold_productos_categorias at all.
# (Products with id_categoria IS NULL are also unresolved — include them.)
unclassified = spark.sql("""
    SELECT DISTINCT sp.id_producto, sp.producto
    FROM workspace.superapp.silver_prices sp
    LEFT JOIN workspace.superapp.gold_productos_categorias pc
        ON sp.id_producto = pc.id_producto
    WHERE (pc.id_producto IS NULL OR pc.id_categoria IS NULL)
      AND sp.producto IS NOT NULL
""").toPandas()

print(f"Unclassified products to process: {len(unclassified):,}")

# COMMAND ----------

# Pre-load up to 5 example product names per category for LLM context.
# This is the "memory" — the LLM always knows what each category looks like.
cat_examples_pd = spark.sql("""
    SELECT gc.nombre, sp.producto
    FROM workspace.superapp.gold_productos_categorias pc
    JOIN workspace.superapp.gold_categorias gc ON pc.id_categoria = gc.id_categoria
    JOIN workspace.superapp.silver_prices sp    ON pc.id_producto = sp.id_producto
    WHERE gc.nombre != 'sin_clasificar'
      AND sp.producto IS NOT NULL
""").toPandas()

cat_examples = {}
for cat, grp in cat_examples_pd.groupby('nombre'):
    cat_examples[cat] = grp['producto'].drop_duplicates().head(5).tolist()

print(f"Examples loaded for {len(cat_examples)} categories.")

# COMMAND ----------

def cosine_sim_batch(embeddings, centroid_matrix):
    """Cosine similarity: each embedding vs every centroid. Returns (n, n_cats)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-10)
    return normed @ centroid_matrix.T  # centroids are already normalized


def get_token():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


def ask_llm(product_name, top_category, top_score, examples, all_categories):
    """
    Medium confidence: ask LLM to confirm top_category or suggest a better one.
    Low confidence:    ask LLM to name the right category (existing or new).
    Returns raw LLM text (lowercase, stripped).
    """
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    token = get_token()
    all_cats_str = ", ".join(all_categories[:60])

    if top_score >= CONFIDENCE_LLM:
        examples_str = "\n".join(f"  - {e}" for e in examples)
        prompt = (
            f'Producto de supermercado argentino: "{product_name}"\n\n'
            f'Categoría candidata: "{top_category}" (similitud={top_score:.2f})\n'
            f'Ejemplos de esa categoría:\n{examples_str}\n\n'
            f'¿Pertenece este producto a "{top_category}"?\n'
            f'Responde EXACTAMENTE con:\n'
            f'  "SI" si pertenece\n'
            f'  "NO: <categoria>" con la categoría correcta (existente o nueva en snake_case español)\n\n'
            f'Categorías existentes: {all_cats_str}\n\nRespuesta:'
        )
    else:
        prompt = (
            f'Producto de supermercado argentino: "{product_name}"\n\n'
            f'No encontré categoría similar (mejor: "{top_category}", similitud={top_score:.2f})\n\n'
            f'Categorías existentes: {all_cats_str}\n\n'
            f'¿A qué categoría pertenece? Responde SOLO con el nombre '
            f'(existente o nuevo en snake_case español).\n\nRespuesta:'
        )

    try:
        resp = requests.post(
            f"https://{workspace_url}/serving-endpoints/{LLM_ENDPOINT}/invocations",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 30, "temperature": 0.1},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        print(f"    LLM error: {str(e)[:80]}")
    return None


def parse_llm_answer(answer, top_category, all_categories):
    """
    Interprets the LLM's answer.
    Returns (resolved_category, is_new_category).
    """
    if answer is None:
        return top_category, False

    if answer == "si":
        return top_category, False

    # "no: <category>" pattern
    if answer.startswith("no:"):
        suggested = answer[3:].strip()
    else:
        suggested = answer

    # Check exact match in known categories
    if suggested in all_categories:
        return suggested, False

    # Partial match
    for cat in all_categories:
        if cat in suggested or suggested in cat:
            return cat, False

    # No match → new category proposal
    return suggested, True

# COMMAND ----------

# ── Main classification loop ──────────────────────────────────────────────────
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

auto_assigned = []   # will be written to gold_productos_categorias
review_queue  = []   # will be written to gold_review_queue

total = len(unclassified)
print(f"Starting classification of {total:,} products...\n")

for batch_start in range(0, total, BATCH_SIZE):
    batch = unclassified.iloc[batch_start : batch_start + BATCH_SIZE]

    # 1. Embed
    embs = model.encode(batch['producto'].tolist(), batch_size=128, show_progress_bar=False)

    # 2. Similarity vs all centroids
    sims      = cosine_sim_batch(embs, centroid_matrix)   # (batch_size, n_cats)
    top_idx   = sims.argmax(axis=1)
    top_scores = sims.max(axis=1)

    for i, (_, row) in enumerate(batch.iterrows()):
        score    = float(top_scores[i])
        cat_name = category_names[top_idx[i]]
        id_cat   = cat_id_map.get(cat_name)

        if score >= CONFIDENCE_AUTO:
            # ── High confidence: auto-assign ─────────────────────────────────
            auto_assigned.append({
                'id_producto':        row['id_producto'],
                'id_categoria':       id_cat,
                'id_subcategoria':    None,
                'metodo':             'embedding_auto',
                'confianza':          round(score, 4),
                'fecha_asignacion':   datetime.now(),
                'usuario_asignacion': 'ml_sistema',
                'notas':              f'centroid_sim={score:.3f}|cat={cat_name}',
            })
        else:
            # ── Medium / low confidence: ask LLM ─────────────────────────────
            examples  = cat_examples.get(cat_name, [])
            answer    = ask_llm(row['producto'], cat_name, score, examples, category_names)
            resolved, is_new = parse_llm_answer(answer, cat_name, category_names)

            if not is_new and resolved in cat_id_map and score >= CONFIDENCE_LLM:
                # LLM confirmed an existing category with medium confidence
                # → still send to review (lower burden, but human validates)
                reason = 'confirmacion_llm'
            elif not is_new and resolved in cat_id_map:
                reason = 'confirmacion_llm'
            else:
                reason = 'categoria_nueva'

            review_queue.append({
                'id_producto':    row['id_producto'],
                'producto':       row['producto'],
                'top_categoria':  cat_name,
                'similitud':      round(score, 4),
                'llm_sugerencia': resolved,
                'es_categoria_nueva': is_new,
                'razon':          reason,
                'estado':         'pendiente',
                'fecha_creacion': datetime.now(),
            })

    pct = min(100, (batch_start + len(batch)) * 100 // total)
    print(f"  [{pct:3d}%] auto={len(auto_assigned):,}  queue={len(review_queue):,}")

print(f"\n=== DONE ===")
print(f"  Auto-assigned:  {len(auto_assigned):,}")
print(f"  Review queue:   {len(review_queue):,}")
print(f"  Total:          {len(auto_assigned) + len(review_queue):,}")

# COMMAND ----------

# Write auto-assigned to gold_productos_categorias (append only).
if auto_assigned:
    (spark.createDataFrame(auto_assigned)
          .write.mode("append")
          .option("mergeSchema", "true")
          .saveAsTable("workspace.superapp.gold_productos_categorias"))
    print(f"Appended {len(auto_assigned):,} rows to gold_productos_categorias")

# COMMAND ----------

# Write review queue to gold_review_queue (append — preserves previous sessions).
if review_queue:
    (spark.createDataFrame(review_queue)
          .write.mode("append")
          .option("mergeSchema", "true")
          .saveAsTable("workspace.superapp.gold_review_queue"))
    print(f"Appended {len(review_queue):,} rows to gold_review_queue")

# COMMAND ----------

# Summary stats
spark.sql("""
    SELECT razon, es_categoria_nueva, estado, COUNT(*) as total
    FROM workspace.superapp.gold_review_queue
    GROUP BY razon, es_categoria_nueva, estado
    ORDER BY total DESC
""").show()
