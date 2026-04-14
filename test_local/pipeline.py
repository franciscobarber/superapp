"""
SuperApp — Local Classification Pipeline
=========================================
Ported from SuperApp/gold_setup/09_convergence_loop.py.ipynb

Run:   python pipeline.py
View:  mlflow ui --backend-store-uri file:///C:/Users/2371180/OneDrive - Cognizant/Documents/test_output/mlruns

Key design decisions (matching notebook 09):
  - Staging → validate → promote  (never writes directly to gold)
  - Adaptive thresholds           (MACRO_AUTO adjusts based on auto-rate per iteration)
  - Rebuild centroids every N iterations from newly labeled products
  - 3 classification paths:
      PATH 3  macro ≥ MACRO_AUTO  AND  cat ≥ CATEGORY_AUTO  → auto-assign
      PATH 2  LLM_MIN ≤ macro < MACRO_AUTO                  → LLM confirms category (if USE_LLM)
      PATH 1  macro < LLM_MIN                               → LLM decides macro first (if USE_LLM)
  - MLflow tracks every iteration (params, metrics, artifacts)
  - Validation spot-checks before promoting to parquet "gold"
"""

import json
import time
import warnings
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Corporate network: inject Windows certificate store so HuggingFace SSL works.
# If truststore is not installed: pip install truststore
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # not on a corporate machine — standard certs are fine

try:
    from sentence_transformers import SentenceTransformer
    import mlflow
except ImportError:
    raise SystemExit(
        "Missing dependencies:\n"
        "  pip install pandas sentence-transformers mlflow pyarrow requests truststore"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors notebook 09 cell 3)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(r"C:\Users\2371180\OneDrive - Cognizant\Documents\sepa_raw")
OUTPUT_DIR = DATA_DIR.parent / "test_output"
MLFLOW_DIR = OUTPUT_DIR / "mlruns"

SILVER_SAMPLE  = 10_000
BATCH_SIZE     = 100           # products per iteration (matches notebook 09)
MAX_ITERATIONS = 30            # 30 × 100 = 3 000 products max
REBUILD_CENTROIDS_EVERY = 5    # rebuild centroids every N iterations

MACRO_AUTO    = 0.90           # initial — adapts per iteration
CATEGORY_AUTO = 0.75

USE_LLM      = False           # set True + provide DATABRICKS_URL/TOKEN to enable
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
LLM_MACRO_MIN = 0.75           # below this → LLM decides macro (path 1)
LLM_MACRO_MAX = 0.90           # above this → embedding decides (path 3)
LLM_TOP_CANDIDATES = 3
LLM_MIN_CONFIDENCE = 0.70      # LLM responses below this go to review queue

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # same as notebook 09
MIN_ACCURACY_PCT = 80.0        # validation threshold before promoting

DATABRICKS_URL   = ""          # e.g. "https://adb-xxx.azuredatabricks.net"
DATABRICKS_TOKEN = ""          # personal access token

TARGET_BARRIOS   = ["monserrat", "once", "balvanera"]
TARGET_LOCALIDAD = ["ciudad autónoma de buenos aires",
                    "ciudad autonoma de buenos aires",
                    "capital federal"]

# ── Macro hierarchy ───────────────────────────────────────────────────────────
MACRO_MAP = {
    "alimentos_basicos":  ["arroz", "aceite", "fideos", "azucar", "sal",
                           "vinagre", "harina", "mayonesa", "mermelada"],
    "lacteos":            ["leche", "queso", "yogur", "manteca"],
    "bebidas":            ["agua", "gaseosa", "jugo", "cerveza", "vino"],
    "infusiones":         ["yerba", "cafe", "te"],
    "panificados_dulces": ["pan", "galletitas", "chocolate"],
    "higiene_personal":   ["jabon", "shampoo", "desodorante", "papel", "crema"],
    "limpieza_hogar":     ["detergente"],
    "mascotas":           ["alimento_gatos", "alimento_perros"],
    "otros":              [],
}
CATEGORY_TO_MACRO = {
    cat: macro for macro, cats in MACRO_MAP.items() for cat in cats
}

# ── Keyword seed ──────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "arroz":       ["arroz"],
    "aceite":      ["aceite"],
    "fideos":      ["fideos", "pasta ", "tallar", "spaghetti"],
    "leche":       ["leche"],
    "yerba":       ["yerba"],
    "azucar":      ["azucar", "azúcar"],
    "jabon":       ["jabon", "jabón", "detergente"],
    "galletitas":  ["galleta", "galletita"],
    "pan":         ["pan lactal", "pan de mi", "panificad"],
    "cafe":        ["cafe ", "café", "nescafe"],
    "te":          ["te en saq", "te herbal", "infusion"],
    "gaseosa":     ["gaseosa", "coca", "pepsi", "sprite", "fanta"],
    "agua":        ["agua mineral", "agua sin gas", "agua con gas"],
    "jugo":        ["jugo", "zumo"],
    "cerveza":     ["cerveza"],
    "vino":        ["vino "],
    "manteca":     ["manteca", "margarina"],
    "queso":       ["queso"],
    "yogur":       ["yogur", "yoghurt"],
    "harina":      ["harina"],
    "shampoo":     ["shampoo", "champú", "champu"],
    "papel":       ["papel higienico", "papel cocina", "servilleta"],
    "desodorante": ["desodorante"],
    "chocolate":   ["chocolate"],
    "mermelada":   ["mermelada", "dulce de"],
    "mayonesa":    ["mayonesa"],
    "sal":         ["sal fina", "sal gruesa", "sal entrefina"],
    "vinagre":     ["vinagre"],
}

# ── Validation spot-check rules (mirrors notebook 09 validate_staging_table) ──
VALIDATION_RULES = [
    (["vino", "wine"],    "vino"),
    (["cerveza", "beer"], "cerveza"),
    (["whisky", "whiskey"], None),   # None = should not be any known category
    (["leche"],           "leche"),
    (["arroz"],           "arroz"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(embs: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / (norms + 1e-10)
    return normed @ matrix.T


def build_centroids(df: pd.DataFrame, group_col: str) -> tuple[np.ndarray, list[str]]:
    names, vecs = [], []
    for name, grp in df.groupby(group_col):
        v    = np.array(grp["embedding"].tolist()).mean(axis=0)
        norm = np.linalg.norm(v)
        vecs.append(v / norm if norm > 1e-10 else v)
        names.append(name)
    return np.array(vecs), names


def keyword_classify(name: str) -> str | None:
    if not name:
        return None
    lower = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return cat
    return None


def load_gold(out: Path) -> pd.DataFrame:
    path = out / "gold_productos_categorias.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  Resuming: {len(df):,} products already in gold")
        return df
    return pd.DataFrame(columns=["id_producto", "producto", "categoria",
                                  "macro", "confianza", "metodo", "iteration", "run_id"])


def save_gold(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "gold_productos_categorias.parquet", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# SILVER
# ─────────────────────────────────────────────────────────────────────────────

def build_silver(data_dir: Path, sample: int | None) -> pd.DataFrame:
    date_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not date_dirs:
        raise FileNotFoundError(f"No date folders in {data_dir}")

    date_dir  = date_dirs[-1]
    week_date = date_dir.name
    print(f"\n[Load] {week_date}")

    def read(fname):
        df = pd.read_csv(date_dir / fname, sep="|", dtype=str, encoding="utf-8")
        df["week_date"] = week_date
        print(f"  {fname:20s}  {len(df):>10,} rows")
        return df

    df_p = read("productos.csv")
    df_s = read("sucursales.csv")
    df_c = read("comercio.csv")

    df_s["_b"] = df_s["sucursales_barrio"].str.lower().str.strip()
    df_s["_l"] = df_s["sucursales_localidad"].str.lower().str.strip()
    local = df_s[df_s["_b"].isin(TARGET_BARRIOS) | df_s["_l"].isin(TARGET_LOCALIDAD)].copy()

    c_dedup = df_c.drop_duplicates("id_comercio")[["id_comercio", "comercio_bandera_nombre"]]
    local   = local.merge(c_dedup, on="id_comercio", how="left")
    counts  = df_p.groupby(["id_sucursal", "week_date"])["id_producto"].count().reset_index(name="n")
    local   = local.merge(counts, on=["id_sucursal", "week_date"], how="inner")
    local["rank"] = (local.groupby(["id_comercio", "_b", "week_date"])["n"]
                         .rank(method="first", ascending=False))
    best = local[local["rank"] == 1]

    silver = df_p.merge(
        best[["id_sucursal", "week_date", "id_comercio",
              "sucursales_barrio", "sucursales_nombre",
              "comercio_bandera_nombre"]],
        on=["id_sucursal", "week_date"], how="inner"
    )
    silver["precio"] = (silver["productos_precio_lista"]
                        .str.replace(",", ".", regex=False)
                        .pipe(pd.to_numeric, errors="coerce"))
    silver = silver[silver["precio"] > 0].rename(columns={
        "productos_descripcion":                "producto",
        "productos_unidad_medida_presentacion": "presentacion",
        "comercio_bandera_nombre":              "cadena",
        "sucursales_nombre":                    "sucursal_nombre",
        "sucursales_barrio":                    "barrio",
    })

    if sample and len(silver) > sample:
        silver = silver.sample(sample, random_state=42).reset_index(drop=True)

    print(f"  Silver: {len(silver):,} rows  |  {silver['id_producto'].nunique():,} unique products")
    return silver


# ─────────────────────────────────────────────────────────────────────────────
# LLM  (optional — mirrors notebook 09 cells 4 + 8)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> dict | None:
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        return None
    try:
        resp = requests.post(
            f"{DATABRICKS_URL}/serving-endpoints/{LLM_ENDPOINT}/invocations",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 60, "temperature": 0.1},
            timeout=30
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # strip markdown code fences if present
            text = text.strip("`").replace("json", "").strip()
            return json.loads(text)
    except Exception as e:
        print(f"    LLM error: {str(e)[:60]}")
    return None


def build_macro_prompt(producto: str, macro_names: list[str]) -> str:
    macros = ", ".join(macro_names)
    return (
        f'Eres un clasificador experto de productos de supermercado.\n\n'
        f'Producto: "{producto}"\n\n'
        f'Macros disponibles: {macros}\n\n'
        f'¿A qué macro pertenece? Responde SOLO con JSON:\n'
        f'{{"macro": "nombre_del_macro", "confianza": 0.0}}\n\n'
        f'Si no estás seguro usa "ninguna".'
    )


def build_category_prompt(producto: str, candidates: list[tuple]) -> str:
    cands = "\n".join(f"- {cat} (similitud={sim:.2f})" for cat, sim, _ in candidates)
    return (
        f'Eres un clasificador experto de productos de supermercado.\n\n'
        f'Producto: "{producto}"\n\n'
        f'Categorías candidatas:\n{cands}\n\n'
        f'¿A qué categoría pertenece? Responde SOLO con JSON:\n'
        f'{{"categoria": "nombre", "confianza": 0.0}}\n\n'
        f'Si no estás seguro usa "ninguna".'
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFY BATCH  (mirrors notebook 09 cell 7 classify_batch_phase1)
# ─────────────────────────────────────────────────────────────────────────────

def classify_batch(batch: pd.DataFrame, model, macro_mat: np.ndarray,
                   macro_names: list, cat_mat: np.ndarray, cat_names: list,
                   macro_auto: float, iteration: int, run_id: str,
                   cat_examples: dict) -> tuple[list, list]:
    """
    Returns (auto_assigned, review_queue) — mirrors 3-path logic from notebook 09.
    """
    auto_assigned = []
    review_queue  = []

    embs         = model.encode(batch["producto"].tolist(), batch_size=128, show_progress_bar=False)

    macro_sims   = cosine_sim(embs, macro_mat)
    macro_top    = macro_sims.argmax(axis=1)
    macro_scores = macro_sims.max(axis=1)

    cat_sims     = cosine_sim(embs, cat_mat)
    cat_top      = cat_sims.argmax(axis=1)
    cat_scores   = cat_sims.max(axis=1)

    for i, (_, row) in enumerate(batch.iterrows()):
        pid         = row["id_producto"]
        produto_val = row["producto"]

        m_score  = float(macro_scores[i])
        c_score  = float(cat_scores[i])
        macro    = macro_names[macro_top[i]]
        cat      = cat_names[cat_top[i]]

        # ── PATH 3: high confidence ────────────────────────────────────────────
        if m_score >= macro_auto and c_score >= CATEGORY_AUTO:
            auto_assigned.append({
                "id_producto":  pid,
                "producto":     produto_val,
                "categoria":   cat,
                "macro":       macro,
                "confianza":   round((m_score + c_score) / 2, 4),
                "metodo":      "embedding_hierarchical_convergence",
                "iteration":   iteration,
                "run_id":      run_id,
                "notas":       f"macro={macro}({m_score:.3f})|cat={cat}({c_score:.3f})",
            })

        # ── PATH 2: medium macro confidence — LLM confirms category ───────────
        elif USE_LLM and LLM_MACRO_MIN <= m_score < macro_auto:
            top_indices  = cat_sims[i].argsort()[-LLM_TOP_CANDIDATES:][::-1]
            candidates   = [(cat_names[j], float(cat_sims[i][j]),
                             cat_examples.get(cat_names[j], [])) for j in top_indices]
            resp = call_llm(build_category_prompt(produto_val, candidates))

            if resp and resp.get("categoria", "ninguna") != "ninguna" \
                    and float(resp.get("confianza", 0)) >= LLM_MIN_CONFIDENCE:
                llm_cat  = resp["categoria"]
                llm_conf = float(resp["confianza"])
                if llm_cat in cat_names:
                    auto_assigned.append({
                        "id_producto": pid, "producto": produto_val,
                        "categoria": llm_cat, "macro": macro,
                        "confianza": round(llm_conf, 4),
                        "metodo": "embedding_hierarchical_llm_convergence",
                        "iteration": iteration, "run_id": run_id,
                        "notas": f"emb_macro={macro}({m_score:.3f})|llm_cat={llm_cat}({llm_conf:.3f})",
                    })
                else:
                    review_queue.append(_queue_row(pid, produto_val, macro, cat,
                                                    m_score, c_score, "llm_cat_not_found"))
            else:
                review_queue.append(_queue_row(pid, produto_val, macro, cat,
                                                m_score, c_score, "llm_category_low_confidence"))

        # ── PATH 1: low macro confidence — LLM decides macro first ────────────
        elif USE_LLM and m_score < LLM_MACRO_MIN:
            resp = call_llm(build_macro_prompt(produto_val, macro_names))

            if resp and resp.get("macro", "ninguna") != "ninguna" \
                    and float(resp.get("confianza", 0)) >= LLM_MIN_CONFIDENCE:
                llm_macro     = resp["macro"]
                llm_macro_conf = float(resp["confianza"])
                # find best category within LLM-selected macro
                macro_cat_mask = [CATEGORY_TO_MACRO.get(c, "otros") == llm_macro for c in cat_names]
                if any(macro_cat_mask):
                    sub_sims = cat_sims[i][macro_cat_mask]
                    sub_names = [c for c, m in zip(cat_names, macro_cat_mask) if m]
                    best_sub  = sub_names[sub_sims.argmax()]
                    best_score = float(sub_sims.max())
                    if best_score >= 0.60:
                        auto_assigned.append({
                            "id_producto": pid, "producto": produto_val,
                            "categoria": best_sub, "macro": llm_macro,
                            "confianza": round((llm_macro_conf + best_score) / 2, 4),
                            "metodo": "embedding_hierarchical_llm_full_convergence",
                            "iteration": iteration, "run_id": run_id,
                            "notas": f"llm_macro={llm_macro}({llm_macro_conf:.3f})|emb_cat={best_sub}({best_score:.3f})",
                        })
                    else:
                        review_queue.append(_queue_row(pid, produto_val, llm_macro, best_sub,
                                                        llm_macro_conf, best_score, "llm_macro_ok_low_category"))
                else:
                    review_queue.append(_queue_row(pid, produto_val, llm_macro, cat,
                                                    llm_macro_conf, c_score, "llm_macro_no_categories"))
            else:
                review_queue.append(_queue_row(pid, produto_val, macro, cat,
                                                m_score, c_score, "llm_macro_low_confidence"))

        # ── No LLM configured — queue everything below threshold ──────────────
        else:
            reason = "low_macro" if m_score < macro_auto else "low_category"
            review_queue.append(_queue_row(pid, produto_val, macro, cat,
                                            m_score, c_score, reason))

    return auto_assigned, review_queue


def _queue_row(pid, produto, macro, cat, m_score, c_score, reason):
    return {
        "id_producto": pid, "producto": produto,
        "top_macro": macro, "top_categoria": cat,
        "macro_sim": round(m_score, 4), "cat_sim": round(c_score, 4),
        "razon": reason, "estado": "pendiente",
        "fecha_creacion": datetime.now(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION  (mirrors notebook 09 validate_staging_table)
# ─────────────────────────────────────────────────────────────────────────────

def validate_staging(staging: pd.DataFrame) -> tuple[bool, float]:
    """
    Spot-checks classifications using VALIDATION_RULES.
    Returns (passed, accuracy_pct).
    """
    if len(staging) == 0:
        return True, 100.0

    errors = 0
    for keywords, expected_cat in VALIDATION_RULES:
        mask = staging["producto"].str.lower().apply(
            lambda n: any(kw in n for kw in keywords) if isinstance(n, str) else False
        )
        subset = staging[mask]
        if len(subset) == 0:
            continue
        if expected_cat is None:
            continue  # no expected category to validate against
        wrong = subset[subset["categoria"] != expected_cat]
        if len(wrong):
            print(f"  Validation: {len(wrong)} '{keywords[0]}' products not in '{expected_cat}'")
            errors += len(wrong)

    accuracy = max(0.0, (len(staging) - errors) / len(staging) * 100)
    passed   = accuracy >= MIN_ACCURACY_PCT
    print(f"  Validation: {len(staging)} classified  |  {errors} errors  |  {accuracy:.1f}% accuracy  → {'PASS' if passed else 'FAIL'}")
    return passed, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
    mlflow.set_experiment("superapp_classification")

    print("=" * 65)
    print("SuperApp Classification Pipeline  (notebook 09 port)")
    print("=" * 65)
    print(f"  Model          : {EMBEDDING_MODEL}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Max iterations : {MAX_ITERATIONS}")
    print(f"  MACRO_AUTO     : {MACRO_AUTO}  (adaptive)")
    print(f"  CATEGORY_AUTO  : {CATEGORY_AUTO}")
    print(f"  Rebuild every  : {REBUILD_CENTROIDS_EVERY} iters")
    print(f"  USE_LLM        : {USE_LLM}")

    silver       = build_silver(DATA_DIR, SILVER_SAMPLE)
    all_products = silver[["id_producto", "producto"]].drop_duplicates("id_producto").copy()
    total = len(all_products)

    gold         = load_gold(OUTPUT_DIR)
    done_ids     = set(gold["id_producto"].tolist())
    unclassified = all_products[~all_products["id_producto"].isin(done_ids)].copy()
    print(f"\n  To classify: {len(unclassified):,} / {total:,}")

    if len(unclassified) == 0:
        print("\nAll products classified. Delete test_output/ to restart.")
        _print_summary(gold, pd.DataFrame(), unclassified)
        return

    print(f"\n[Model] Loading {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    run_name = f"convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    staging       = []    # in-memory staging (mirrors notebook 09 staging table)
    review_rows   = []

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Run: {run_id[:8]}")

        mlflow.log_params({
            "embedding_model":        EMBEDDING_MODEL,
            "batch_size":             BATCH_SIZE,
            "max_iterations":         MAX_ITERATIONS,
            "macro_auto_initial":     MACRO_AUTO,
            "category_auto":          CATEGORY_AUTO,
            "rebuild_centroids_every": REBUILD_CENTROIDS_EVERY,
            "use_llm":                USE_LLM,
            "silver_sample":          SILVER_SAMPLE,
        })

        # ── Iter 0: keyword rules ─────────────────────────────────────────────
        unclassified["categoria"] = unclassified["producto"].apply(keyword_classify)
        new_kw = unclassified[unclassified["categoria"].notna()].copy()
        new_kw["macro"]     = new_kw["categoria"].map(lambda c: CATEGORY_TO_MACRO.get(c, "outros"))
        new_kw["confianza"] = 0.80
        new_kw["metodo"]    = "keyword_rule"
        new_kw["iteration"] = 0
        new_kw["run_id"]    = run_id
        new_kw["notas"]     = "keyword_seed"

        staging.extend(new_kw[["id_producto", "producto", "categoria", "macro",
                                "confianza", "metodo", "iteration", "run_id", "notas"]].to_dict("records"))
        done_ids.update(new_kw["id_producto"].tolist())
        unclassified = unclassified[unclassified["categoria"].isna()][["id_producto", "producto"]].copy()

        mlflow.log_metrics({"iter_0_new": len(new_kw), "iter_0_remaining": len(unclassified)})
        print(f"\n[Iter 0] keyword rules → +{len(new_kw):,}  |  {len(unclassified):,} remaining")

        # ── Iter 1+: embedding convergence loop ───────────────────────────────
        current_macro_auto = MACRO_AUTO
        all_labeled        = new_kw.copy()   # grows each iteration
        iteration_results  = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            if len(unclassified) == 0:
                print("\nConverged — all products classified.")
                break

            print(f"\n[Iter {iteration:2d}]  unclassified={len(unclassified):,}  "
                  f"macro_auto={current_macro_auto:.2f}")

            # Rebuild centroids every N iters or on first iter
            if iteration == 1 or (iteration % REBUILD_CENTROIDS_EVERY == 0):
                print(f"  Rebuilding centroids from {len(all_labeled):,} labeled products ...")
                all_labeled["macro_for_centroid"] = all_labeled["categoria"].map(
                    lambda c: CATEGORY_TO_MACRO.get(c, "outros")
                )
                embs_lab = model.encode(all_labeled["producto"].tolist(),
                                        batch_size=128, show_progress_bar=True)
                all_labeled = all_labeled.copy()
                all_labeled["embedding"] = list(embs_lab)

                macro_mat, macro_names = build_centroids(all_labeled, "macro_for_centroid")
                cat_mat,   cat_names   = build_centroids(all_labeled, "categoria")

                # Pre-load examples per category for LLM prompts
                cat_examples = {}
                for cat, grp in all_labeled.groupby("categoria"):
                    cat_examples[cat] = grp["producto"].head(3).tolist()

                print(f"  Centroids: {len(macro_names)} macros, {len(cat_names)} categories")

                # Save centroid names as MLflow artifact
                c_path = OUTPUT_DIR / f"centroids_iter{iteration}.json"
                c_path.write_text(json.dumps({"macros": macro_names, "categories": cat_names}))
                mlflow.log_artifact(str(c_path))

            # Sample a batch
            batch = (unclassified.sample(min(BATCH_SIZE, len(unclassified)), random_state=iteration)
                     .copy())

            new_auto, new_queue = classify_batch(
                batch, model, macro_mat, macro_names,
                cat_mat, cat_names, current_macro_auto,
                iteration, run_id, cat_examples
            )

            auto_rate = len(new_auto) / max(len(batch), 1)

            # ── Adaptive threshold (matches notebook 09) ──────────────────────
            if auto_rate < 0.40:
                current_macro_auto = max(0.65, current_macro_auto - 0.05)
            elif auto_rate > 0.60:
                current_macro_auto = min(0.90, current_macro_auto + 0.05)

            # Update state
            if new_auto:
                new_df = pd.DataFrame(new_auto)
                staging.extend(new_auto)
                done_ids.update(new_df["id_producto"].tolist())
                unclassified = unclassified[~unclassified["id_producto"].isin(done_ids)].copy()
                all_labeled  = pd.concat([all_labeled, new_df[["id_producto", "producto", "categoria"]]],
                                         ignore_index=True)

            review_rows.extend(new_queue)

            mlflow.log_metrics({
                f"iter_{iteration}_new":       len(new_auto),
                f"iter_{iteration}_queued":    len(new_queue),
                f"iter_{iteration}_macro_auto": current_macro_auto,
                f"iter_{iteration}_auto_rate": round(auto_rate, 3),
            }, step=iteration)

            iteration_results.append({
                "iteration": iteration, "batch": len(batch),
                "new_auto": len(new_auto), "queued": len(new_queue),
                "auto_rate": round(auto_rate * 100, 1),
                "macro_auto": round(current_macro_auto, 2),
                "remaining": len(unclassified),
            })

            print(f"  +{len(new_auto)} auto  |  {len(new_queue)} queued  "
                  f"|  auto_rate={auto_rate*100:.0f}%  |  {len(unclassified):,} remaining")

            if len(new_auto) == 0 and not USE_LLM:
                print(f"  No new assignments — converged at iteration {iteration}.")
                mlflow.log_metric("converged_at", iteration)
                break

        # ── Staging → validate → promote ─────────────────────────────────────
        staging_df = pd.DataFrame(staging)
        print(f"\n[Validate] {len(staging_df):,} classifications in staging ...")
        passed, accuracy = validate_staging(staging_df)

        mlflow.log_metric("validation_accuracy", round(accuracy, 2))

        if passed:
            # Promote: merge staging with existing gold
            promoted = pd.concat([gold, staging_df], ignore_index=True)
            promoted  = promoted.drop_duplicates(subset=["id_producto"], keep="last")
            save_gold(promoted, OUTPUT_DIR)
            mlflow.log_metric("promoted_count", len(staging_df))
            print(f"  PROMOTED {len(staging_df):,} → gold_productos_categorias.parquet")
        else:
            # Save staging separately for manual review
            staging_df.to_parquet(OUTPUT_DIR / f"staging_{run_id[:8]}.parquet", index=False)
            print(f"  FAILED validation — saved staging_{run_id[:8]}.parquet for review")

        # ── Save review queue ─────────────────────────────────────────────────
        if review_rows:
            q_df      = pd.DataFrame(review_rows)
            q_path    = OUTPUT_DIR / "review_queue.parquet"
            if q_path.exists():
                prev = pd.read_parquet(q_path)
                prev = prev[~prev["id_producto"].isin(done_ids)]
                q_df = pd.concat([prev, q_df], ignore_index=True)
            q_df.to_parquet(q_path, index=False)

        # ── Final silver join ─────────────────────────────────────────────────
        if passed:
            gold_final = pd.read_parquet(OUTPUT_DIR / "gold_productos_categorias.parquet")
            silver_out = silver.merge(
                gold_final[["id_producto", "categoria", "macro", "confianza", "metodo"]],
                left_on="id_producto", right_on="id_producto", how="left"
            )
            silver_out.to_parquet(OUTPUT_DIR / "silver_classified.parquet", index=False)

        silver.to_parquet(OUTPUT_DIR / "silver_prices.parquet", index=False)
        mlflow.log_metric("final_macro_auto", current_macro_auto)

    _print_summary(
        pd.read_parquet(OUTPUT_DIR / "gold_productos_categorias.parquet") if passed else staging_df,
        pd.DataFrame(review_rows),
        unclassified,
        iteration_results
    )


def _print_summary(gold, queue, unclassified, iters=None):
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Gold classified : {len(gold):,}")
    print(f"  Review queue    : {len(queue):,}")
    print(f"  Unclassified    : {len(unclassified):,}")
    if len(gold):
        print("\n  Method breakdown:")
        print(gold["metodo"].value_counts().to_string())
        print("\n  Top 15 categories:")
        print(gold["categoria"].value_counts().head(15).to_string())
    if iters:
        print("\n  Iteration summary:")
        print(pd.DataFrame(iters).to_string(index=False))
    print(f"\n  Output : {OUTPUT_DIR}")
    print(f"  MLflow : mlflow ui --backend-store-uri file:///{MLFLOW_DIR}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
