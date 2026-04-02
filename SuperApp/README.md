# SuperApp — Supermarket Price Tracker

Tracks weekly supermarket prices from the Argentine government's SEPA dataset
(`datos.produccion.gob.ar/dataset/sepa-precios`), stores them in Delta tables,
and categorizes products so prices can be compared across brands over time.

---

## Folder structure

```
SuperApp/
├── explorations/          Ad-hoc notebooks for data exploration
├── transformations/       Lakeflow pipeline (Bronze → Silver)
├── gold_setup/            One-time setup and ML categorization notebooks
└── utilities/             Shared UDFs and helpers
```

---

## Architecture

```
Government website (SEPA Precios — every Thursday)
        │
[01_downloader.py]          Standalone notebook (run before pipeline)
  Scrapes Thursday URLs, downloads ZIPs, writes CSVs to Volume
        │
        ▼
/Volumes/workspace/superapp/sepa_raw/YYYY-MM-DD/
  ├── productos.csv
  ├── sucursales.csv
  └── comercio.csv
        │
[Lakeflow ETL Pipeline]     transformations/etl_superapp.py
  Bronze tables (Auto Loader, streaming)
    bronze_productos
    bronze_sucursales
    bronze_comercio
        │
  Silver materialized view
    silver_prices           Joined, neighborhood-filtered, price-normalized
        │
[Gold Setup notebooks]      gold_setup/ — run once, then iteratively
  gold_categorias           Category dimension table
  gold_productos_categorias Product → category mapping
  gold_category_centroids   Embedding centroids (ML memory)
  gold_review_queue         Items pending human review
```

**Catalog:** `workspace` | **Schema:** `superapp`
**Neighborhoods tracked:** Balvanera, Once, Monserrat (Buenos Aires)

---

## Pipeline — transformations/

| File | Purpose |
|---|---|
| `etl_superapp.py` | Main Lakeflow pipeline: Bronze streaming tables + Silver materialized view |
| `gold_layer_addition.py` | Gold layer additions to the pipeline |

Run with **Run pipeline** in the Databricks UI. Schedule weekly (Thursdays).

---

## Gold Setup — gold_setup/

Run these notebooks **in order**, once. After the first full run, notebooks 05–07
are re-run iteratively to improve categorization.

| Notebook | What it does | Run |
|---|---|---|
| `01_create_gold_tables.py.ipynb` | Creates `gold_categorias` and `gold_productos_categorias` DDL | Once |
| `02_populate_initial_categories.py.ipynb` | Keyword rules → classifies ~13K products (`metodo='regla'`) | Once |
| `05_build_category_embeddings.py` | Embeds labeled products, computes per-category centroids → `gold_category_centroids` | Once, then after each review session |
| `06_classify_with_embeddings.py` | Classifies unclassified products by centroid similarity + LLM confirmation | Each iteration |
| `07_review_queue.py` | Human review of uncertain items; approve/reject/create categories | Each iteration |

### ML categorization flow

```
Labeled products (gold_productos_categorias)
        │
[05] Compute category centroids  ←─────────────────────────┐
        │                                                   │
        ▼                                                   │ re-run after
[06] For each unclassified product:                        │ approvals
  similarity ≥ 0.85  →  auto-assign (embedding_auto)       │
  0.60 – 0.85        →  LLM confirms  →  review queue      │
  < 0.60             →  LLM proposes new category  →  review queue
        │
[07] Human reviews queue
  approve_batch(ids, "category_name")   ──────────────────────┘
  create_category("name", "description")
  reject_batch(ids)
```

Each iteration increases the auto-assignment rate as centroids improve.

**Models used:**
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, fast)
- LLM confirmation: `databricks-meta-llama-3-3-70b-instruct`

---

## Weekly run order

1. Run `01_downloader.py` (standalone notebook, not part of pipeline)
2. Trigger the **Lakeflow pipeline** (or let the schedule run it)
3. Optionally re-run `06` + `07` if new unclassified products appeared

---

## Setup (first time)

```sql
-- Run in SQL Editor
CREATE SCHEMA IF NOT EXISTS workspace.superapp;
CREATE VOLUME IF NOT EXISTS workspace.superapp.sepa_raw;
```

Then run gold_setup notebooks 01 → 02 → 05 → 06 → 07 in order.