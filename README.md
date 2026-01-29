# HitPromo (ML demo)

This workspace runs a small Flask UI (`hitpromo.py`) for:

- Product recommendations
- Demand forecasting
- Pricing recommendations

All three are **machine-learning driven by default**, using a downloaded sample dataset (MovieLens 100k) that is cached locally.

## Run

```powershell
C:/hitpromo/.venv/Scripts/python.exe hitpromo.py
```

Open: `http://127.0.0.1:8005`

## Data sources

The app looks for local CSVs first:

- `data/products.csv`
- `data/orders.csv`

If `data/orders.csv` is missing, it will try to download MovieLens 100k and convert it into an order-like table.

Control this with:

- `HITPROMO_DATA_SOURCE=movielens` (default) → downloads/caches under `data/sample/movielens-100k/`
- `HITPROMO_DATA_SOURCE=demo` → uses the built-in synthetic generator
- `HITPROMO_DATA_SOURCE=csv` → only use `data/orders.csv` (no sample download)
