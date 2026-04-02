# Databricks Notebook — Gold Setup Step 7
# Human-in-the-loop review of uncertain classifications.
#
# Workflow:
#   1. Run Cell "INSPECT" to see what's pending grouped by LLM suggestion
#   2. For each suggestion group:
#      a. If it maps to an existing category → call approve_batch(ids, "category_name")
#      b. If it's a new valid category       → call create_category(...) then approve_batch(...)
#      c. If it's noise/brand               → call reject_batch(ids)
#   3. After reviewing a session, re-run 05_build_category_embeddings.py
#      to update centroids with newly approved products.
#
# Tables affected:
#   gold_review_queue         (estado updated: pendiente → aprobado / rechazado)
#   gold_productos_categorias (new rows inserted on approval)
#   gold_categorias           (new rows inserted via create_category)

# COMMAND ----------

import pandas as pd
from datetime import datetime
from pyspark.sql.functions import col

# COMMAND ----------
# ── INSPECT ──────────────────────────────────────────────────────────────────
# Run this cell to see what needs reviewing.

pending = spark.sql("""
    SELECT id_producto, producto, top_categoria, similitud,
           llm_sugerencia, es_categoria_nueva, razon, fecha_creacion
    FROM workspace.superapp.gold_review_queue
    WHERE estado = 'pendiente'
    ORDER BY es_categoria_nueva DESC, similitud ASC
""").toPandas()

print(f"Pending review: {len(pending):,} items")
print(f"  Needs new category:  {pending['es_categoria_nueva'].sum():,}")
print(f"  LLM confirmation:    {(~pending['es_categoria_nueva']).sum():,}")

print("\n=== GROUPED BY LLM SUGGESTION ===\n")
summary = (
    pending
    .groupby(['llm_sugerencia', 'es_categoria_nueva'])
    .agg(count=('id_producto', 'count'),
         examples=('producto', lambda x: list(x.head(5))))
    .sort_values('count', ascending=False)
)

for (sugerencia, is_new), row in summary.iterrows():
    tag = "[NEW]" if is_new else "     "
    print(f"{tag} [{row['count']:4d}]  {sugerencia}")
    for ex in row['examples']:
        print(f"               • {ex}")
    print()

# COMMAND ----------
# ── HELPERS ──────────────────────────────────────────────────────────────────

def _mark_queue(product_ids, estado, category_name=None):
    ids_sql = ",".join(f"'{p}'" for p in product_ids)
    cat_sql = f", llm_sugerencia = '{category_name}'" if category_name else ""
    spark.sql(f"""
        UPDATE workspace.superapp.gold_review_queue
        SET estado = '{estado}'{cat_sql}
        WHERE id_producto IN ({ids_sql})
          AND estado = 'pendiente'
    """)


def approve_batch(product_ids, category_name):
    """
    Assign product_ids to an existing category.
    category_name must already exist in gold_categorias.
    """
    cat_row = (spark.table("workspace.superapp.gold_categorias")
               .filter(col("nombre") == category_name)
               .first())
    if not cat_row:
        print(f"  ERROR: category '{category_name}' not found. Use create_category() first.")
        return

    rows = [{
        'id_producto':        pid,
        'id_categoria':       int(cat_row.id_categoria),
        'id_subcategoria':    None,
        'metodo':             'revision_humana',
        'confianza':          1.0,
        'fecha_asignacion':   datetime.now(),
        'usuario_asignacion': 'revisor',
        'notas':              f'aprobado_desde_queue',
    } for pid in product_ids]

    (spark.createDataFrame(rows)
          .write.mode("append")
          .option("mergeSchema", "true")
          .saveAsTable("workspace.superapp.gold_productos_categorias"))

    _mark_queue(product_ids, "aprobado", category_name)
    print(f"  Approved {len(rows)} products → '{category_name}'")


def create_category(nombre, descripcion, parent_id=None):
    """
    Insert a new row in gold_categorias.
    Call this before approve_batch() when the LLM proposed a new category.
    Returns the new id_categoria.
    """
    max_id = (spark.sql("SELECT COALESCE(MAX(id_categoria), 0) AS m FROM workspace.superapp.gold_categorias")
              .first().m)
    new_id = int(max_id) + 1

    parent_sql = str(parent_id) if parent_id else "NULL"
    spark.sql(f"""
        INSERT INTO workspace.superapp.gold_categorias
            (id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion)
        VALUES ({new_id}, '{nombre}', 'categoria', {parent_sql}, '{descripcion}', current_timestamp())
    """)
    print(f"  Created category '{nombre}' (id={new_id})")
    return new_id


def reject_batch(product_ids, assign_sin_clasificar=True):
    """
    Reject items (noise, brands, fragments).
    If assign_sin_clasificar=True, assigns them to the 'sin_clasificar' category
    so they don't re-appear in future runs of notebook 06.
    """
    _mark_queue(product_ids, "rechazado")

    if assign_sin_clasificar:
        sin_cat = (spark.table("workspace.superapp.gold_categorias")
                   .filter(col("nombre") == "sin_clasificar")
                   .first())
        if sin_cat:
            rows = [{
                'id_producto':        pid,
                'id_categoria':       int(sin_cat.id_categoria),
                'id_subcategoria':    None,
                'metodo':             'revision_humana',
                'confianza':          1.0,
                'fecha_asignacion':   datetime.now(),
                'usuario_asignacion': 'revisor',
                'notas':              'rechazado_en_queue',
            } for pid in product_ids]
            (spark.createDataFrame(rows)
                  .write.mode("append")
                  .option("mergeSchema", "true")
                  .saveAsTable("workspace.superapp.gold_productos_categorias"))

    print(f"  Rejected {len(product_ids)} products.")


def get_pending_ids(llm_suggestion):
    """Shortcut: get all pending product_ids for a given LLM suggestion."""
    return (spark.sql(f"""
        SELECT id_producto FROM workspace.superapp.gold_review_queue
        WHERE estado = 'pendiente'
          AND llm_sugerencia = '{llm_suggestion}'
    """).toPandas()['id_producto'].tolist())

print("Helper functions loaded:")
print("  approve_batch(product_ids, category_name)")
print("  create_category(nombre, descripcion, parent_id=None)")
print("  reject_batch(product_ids, assign_sin_clasificar=True)")
print("  get_pending_ids(llm_suggestion)  ← shortcut to fetch ids by suggestion")

# COMMAND ----------
# ── REVIEW SESSION ───────────────────────────────────────────────────────────
# Edit and run this cell for each suggestion group you want to process.
# Example workflow (uncomment and adapt):

# --- Step 1: Create new categories that don't exist yet ---
# create_category("desodorantes",    "Desodorantes y antitranspirantes")
# create_category("protector_solar", "Protectores solares y bloqueadores")
# create_category("televisores",     "Televisores y pantallas")

# --- Step 2: Approve groups ---
# approve_batch(get_pending_ids("desodorantes"),    "desodorantes")
# approve_batch(get_pending_ids("protector_solar"), "protector_solar")
# approve_batch(get_pending_ids("televisores"),     "televisores")

# --- Step 3: Map LLM suggestions to existing categories ---
# approve_batch(get_pending_ids("te_infusion"), "te")      # already exists as "te"
# approve_batch(get_pending_ids("champú"),      "shampoo")

# --- Step 4: Reject noise ---
# reject_batch(get_pending_ids("articulos_arco_iris"))  # brand, not category
# reject_batch(get_pending_ids("productos_johnson"))    # brand

print("Edit the REVIEW SESSION cell above and run it.")

# COMMAND ----------
# ── PROGRESS SUMMARY ─────────────────────────────────────────────────────────

spark.sql("""
    SELECT
        estado,
        COUNT(*)                              AS total,
        ROUND(COUNT(*) * 100.0
              / SUM(COUNT(*)) OVER (), 1)     AS pct,
        COUNT(DISTINCT llm_sugerencia)        AS unique_suggestions
    FROM workspace.superapp.gold_review_queue
    GROUP BY estado
    ORDER BY total DESC
""").show()

# COMMAND ----------
# ── AFTER REVIEW ─────────────────────────────────────────────────────────────
# Once you've approved enough new items, re-run notebook 05 to update centroids:
#   → SuperApp/gold_setup/05_build_category_embeddings.py
# Then re-run notebook 06 to classify remaining unclassified products:
#   → SuperApp/gold_setup/06_classify_with_embeddings.py
# Each iteration the auto-assignment rate should increase as centroids improve.

print("After approvals: re-run 05_build_category_embeddings.py to update centroids.")
print("Then re-run 06_classify_with_embeddings.py for the next iteration.")
