# SuperApp — Claude Instructions

## Receipt OCR → gold_user_purchases

When the user pastes a receipt image, extract all products and return a ready-to-run
Databricks SQL INSERT. No explanation needed — just the table preview and the SQL.

### Target table

```
workspace.superapp.gold_user_purchases (
    ticket_id       STRING,   -- transaction number from receipt footer
    fecha           DATE,     -- purchase date
    id_producto     BIGINT,   -- leave NULL (matched later by barcode)
    producto        STRING,   -- full product name as printed
    precio_pagado   DECIMAL,  -- final amount charged (after discount)
    precio_base     DECIMAL,  -- price before discount (= precio_pagado if no discount)
    descuento       DECIMAL,  -- discount amount, positive (0 if none)
    cantidad        DECIMAL,  -- units or kilograms
    unidad          STRING,   -- "un" or "kg"
    precio_unitario DECIMAL,  -- price per unit or per kg
    barcode_ean13   STRING,   -- 13-digit EAN if visible, else NULL
    id_comercio     INT,      -- 1 = COTO, 2 = Carrefour, 3 = Disco (add as needed)
    source_file     STRING,   -- describe as "chat_YYYY-MM-DD"
    confidence      DECIMAL,  -- 0.97 for clear images, 0.85 for blurry/partial
    created_at      TIMESTAMP -- leave as current_timestamp()
)
```

### COTO receipt parsing rules

COTO Argentina receipts (thermal paper, Spanish) have these patterns:

| Pattern | Example | How to parse |
|---|---|---|
| Simple item | `ACEITE GIRASOL 1.5 LTR  6160,00` | producto + precio_base on same line |
| Weight item | `0,398 x 39999,00` then `LANGOSTINO KG` | precio_pagado = 0.398 × 39999, unidad=kg |
| `=` prefix | `=ROAST BEEF ESTANCIA  11503,64` | final confirmed price |
| Discount line | `-974,70` or `1 *25% CLASES` or `30 OFF PROMO VISA DEBIT` | subtract from precio_base |
| Barcode line | `0000511016 07798144510464` | last 13 digits = barcode_ean13 |
| Internal code | `0000511016` (10 digits) | ignore |

**Key rules:**
- For weight items: `precio_pagado = cantidad × precio_unitario` (calculate it)
- `precio_base` = price before discount; `precio_pagado` = precio_base − descuento
- Lines containing `PROMO VISA`, `CLASES`, `SUBTOTAL`, `TOTAL`, `VISA`, `SU VUELTO`,
  `FACTURA`, `IVA`, `CUIT`, `NRO` → skip entirely
- Receipt may be photographed sideways — rotate mentally 90° if needed
- Barcode starting with `02` = variable-weight product (meat/deli), `07` = packaged product

### id_comercio mapping

| id | Comercio |
|---|---|
| 1 | COTO |
| 2 | Carrefour |
| 3 | Disco |
| 4 | Jumbo |
| 5 | DIA |

Add new ones as encountered.

### Output format

**Step 1** — show a compact table of extracted products:

```
fecha: DD/MM/YYYY  |  comercio: COTO  |  ticket: XXXXXX  |  N products
─────────────────────────────────────────────────────────────────────────
PRODUCTO                          cant  un  p.unitario  p.final  descuento  barcode
ACEITE GIRASOL COCINEROBOT 1.5L   1.00  un    6160.00  6160.00       0.00  07790060023684
LANGOSTINO PELADO CRUDO GRANDEX   0.40  kg   39999.00 15919.60    4775.88  02580592003989
...
─────────────────────────────────────────────────────────────────────────
Total pagado: XX,XXX.XX  |  Total ahorrado: XX,XXX.XX  |  % ahorro: XX%
```

**Step 2** — immediately after the table, output the INSERT SQL:

```sql
INSERT INTO workspace.superapp.gold_user_purchases
  (ticket_id, fecha, id_producto, producto, precio_pagado, precio_base,
   descuento, cantidad, unidad, precio_unitario, barcode_ean13,
   id_comercio, source_file, confidence, created_at)
VALUES
  ('TICKET_ID', '2026-04-16', NULL, 'PRODUCTO 1', 6160.00, 6160.00, 0.00, 1.000, 'un', 6160.00, '07790060023684', 1, 'chat_2026-04-16', 0.97, current_timestamp()),
  ('TICKET_ID', '2026-04-16', NULL, 'PRODUCTO 2', 15919.60, 20695.48, 4775.88, 0.398, 'kg', 39999.00, '02580592003989', 1, 'chat_2026-04-16', 0.97, current_timestamp()),
  -- ... one row per product
;
```

Date format in SQL VALUES must be `'YYYY-MM-DD'` (ISO), not `DD/MM/YYYY`.

### Validation check

After the INSERT, show this query for the user to run and verify:

```sql
SELECT fecha, ticket_id, COUNT(*) AS items,
       ROUND(SUM(precio_pagado),2) AS total,
       ROUND(SUM(descuento),2) AS ahorro,
       ROUND(SUM(descuento)/NULLIF(SUM(precio_pagado+descuento),0)*100,1) AS pct_ahorro
FROM workspace.superapp.gold_user_purchases
WHERE ticket_id = 'TICKET_ID'
GROUP BY fecha, ticket_id;
```

### If a price is missing or unclear

- Set `precio_pagado = NULL`, `confidence = 0.70`
- Add a comment in the SQL: `-- PRICE UNCLEAR, verify manually`
- Do not invent or estimate prices

### Multiple receipts in one message

Process each receipt separately, with its own table + INSERT block.
Label each section with the receipt date and store.
