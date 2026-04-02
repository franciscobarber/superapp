# Databricks Lakeflow Pipeline — Step 2: Bronze → Silver
# Attach this file to a Delta Live Tables pipeline in your workspace.
# Reads CSVs written by 01_downloader.py from the Volume, builds Bronze
# streaming tables and a Silver materialized view.

from pyspark import pipelines as dp
from pyspark.sql.functions import (
    col, lower, to_date, regexp_extract, regexp_replace, count, row_number, when, lit, concat_ws
)
from pyspark.sql.types import DoubleType
from pyspark.sql import Window

# ── Config ───────────────────────────────────────────────────────────────────

CATALOG      = "workspace"
SCHEMA       = "superapp"
VOLUME_BASE  = f"/Volumes/{CATALOG}/{SCHEMA}/sepa_raw"

# Target neighborhoods (case-insensitive) - checking both barrio and locality
TARGET_BARRIOS = ["monserrat", "once", "balvanera"]
TARGET_LOCALITY = ["ciudad autónoma de buenos aires", "ciudad autonoma de buenos aires", "capital federal"]

# Pattern to extract the week date from the Volume path, e.g. .../2024-12-05/productos.csv
DATE_PATTERN = r"(\d{4}-\d{2}-\d{2})"


# ── BRONZE ───────────────────────────────────────────────────────────────────
# Three streaming tables — Auto Loader picks up each new date folder written
# by the downloader. Schema inference is disabled; all columns land as strings
# (safe for raw layer; casting happens in Silver).

@dp.table(comment="Raw product rows from SEPA weekly CSV exports.")
def bronze_productos():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("sep", "|")
            .option("header", "true")
            .option("encoding", "UTF-8")
            .option("cloudFiles.inferColumnTypes", "false")
            .load(f"{VOLUME_BASE}/*/productos.csv")
            .select(
                regexp_extract("_metadata.file_path", DATE_PATTERN, 1).alias("week_date"),
                col("id_comercio"),
                col("id_sucursal"),
                col("id_producto"),
                col("productos_descripcion"),
                col("productos_unidad_medida_presentacion").alias("presentacion"),
                col("productos_precio_lista"),
                col("productos_precio_referencia"),
                col("productos_cantidad_referencia"),
            )
    )


@dp.table(comment="Raw branch rows from SEPA weekly CSV exports.")
def bronze_sucursales():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("sep", "|")
            .option("header", "true")
            .option("encoding", "UTF-8")
            .option("cloudFiles.inferColumnTypes", "false")
            .load(f"{VOLUME_BASE}/*/sucursales.csv")
            .select(
                regexp_extract("_metadata.file_path", DATE_PATTERN, 1).alias("week_date"),
                col("id_sucursal"),
                col("id_comercio"),
                col("sucursales_barrio"),
                col("sucursales_localidad"),
                col("sucursales_calle"),
                col("sucursales_numero"),
                col("sucursales_nombre").alias("sucursal_nombre"),
            )
    )


@dp.table(comment="Raw supermarket chain rows from SEPA weekly CSV exports.")
def bronze_comercio():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("sep", "|")
            .option("header", "true")
            .option("encoding", "UTF-8")
            .option("cloudFiles.inferColumnTypes", "false")
            .load(f"{VOLUME_BASE}/*/comercio.csv")
            .select(
                col("id_comercio"),
                col("comercio_bandera_nombre").alias("cadena"),
            )
    )


# ── SILVER ────────────────────────────────────────────────────────────────────
# Joins all three Bronze tables with smart sucursal selection:
# - One sucursal per comercio per target barrio
# - Multiple sucursales per comercio only if in different target barrios
# - For CABA fallback, one sucursal per comercio with most products

@dp.materialized_view(
    comment=(
        "Cleaned prices with smart sucursal selection: "
        "one store per comercio per target barrio, allows multiple stores per comercio only if in different barrios."
    )
)
@dp.expect("valid_price",        "precio > 0")
@dp.expect("valid_product_name", "producto IS NOT NULL")
@dp.expect("valid_week_date",    "week_date IS NOT NULL")
def silver_prices():
    productos  = spark.read.table("bronze_productos")
    sucursales = spark.read.table("bronze_sucursales")
    comercio   = spark.read.table("bronze_comercio").dropDuplicates(["id_comercio"])
    
    # Join sucursales with comercio early to get cadena
    sucursales_with_comercio = sucursales.join(
        comercio,
        on="id_comercio",
        how="inner"
    )
    
    # Filter sucursales for target location (check both barrio and locality)
    sucursales_filtered = sucursales_with_comercio.filter(
        lower(col("sucursales_barrio")).isin(TARGET_BARRIOS) |
        lower(col("sucursales_localidad")).isin(TARGET_LOCALITY)
    )
    
    # Create barrio_group: use specific barrio name if in target list, otherwise "caba"
    barrio_group_expr = lit("caba")
    for barrio in TARGET_BARRIOS:
        barrio_group_expr = when(
            lower(col("sucursales_barrio")) == barrio,
            lit(barrio)
        ).otherwise(barrio_group_expr)
    
    sucursales_filtered = sucursales_filtered.withColumn(
        "barrio_group",
        barrio_group_expr
    )
    
    # Count products per sucursal (to find most representative store per comercio+barrio)
    product_counts = productos.groupBy("id_sucursal", "week_date").agg(
        count("id_producto").alias("product_count")
    )
    
    sucursales_with_counts = sucursales_filtered.join(
        product_counts,
        on=["id_sucursal", "week_date"],
        how="inner"  # Only keep sucursales that have products
    )
    
    # Rank by product count within each comercio+barrio_group+week
    # This ensures one sucursal per comercio per target barrio, or one per comercio for CABA
    windowSpec = Window.partitionBy("id_comercio", "barrio_group", "week_date").orderBy(col("product_count").desc())
    best_sucursal_per_comercio_barrio = sucursales_with_counts.withColumn(
        "rank",
        row_number().over(windowSpec)
    ).filter(col("rank") == 1)
    
    # Join with productos, normalize price
    result = (
        productos
            .drop("id_comercio")  # Drop to avoid ambiguity
            .join(best_sucursal_per_comercio_barrio, on=["id_sucursal", "week_date"], how="inner")
            # Normalize price: Argentine locale uses comma as decimal separator
            .withColumn(
                "precio",
                regexp_replace(col("productos_precio_lista"), ",", ".").cast(DoubleType())
            )
            .withColumn("week_date", to_date(col("week_date"), "yyyy-MM-dd"))
            .filter(col("precio").isNotNull() & (col("precio") > 0))
    )
    
    # Select columns - cadena already present from early join
    return result.select(
        col("week_date"),
        col("id_producto"),
        col("id_comercio"),
        col("productos_descripcion").alias("producto"),
        col("presentacion"),
        col("cadena"),
        col("sucursal_nombre"),
        col("sucursales_barrio"),
        concat_ws(" ", col("sucursales_calle"), col("sucursales_numero")).alias("sucursales_direccion"),
        col("precio"),
    )
