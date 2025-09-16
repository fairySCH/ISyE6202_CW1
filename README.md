
# ISyE 6202 – Casework 1: Dobda SC Facility Network Analysis

## Repository Structure

```md
CW1/
├── data/                  # Input datasets
│   ├── demand\_seasonalities.csv
│   ├── fc\_zip3\_distance.csv
│   ├── msa.csv
│   ├── zip3\_coordinates.csv
│   ├── zip3\_market.csv
│   ├── zip3\_pmf.csv
│   └── market\_demand.xlsx
├── output/                # All generated results
│   ├── task1\_2/
│   ├── task3/
│   ├── task4/
│   ├── task5/
│   ├── task6/
│   ├── task7/
│   ├── task8/
│   └── task9\_10/
├── src/                   # Computation scripts
│   ├── task1\_2.py
│   ├── task3.py
│   ├── task4.py
│   ├── task5.py
│   ├── task6.py
│   ├── Task7.py
│   └── Task8.py
└── report/                # Final report document(s)
```

## Execution Flow
1. **Task 1–2 (`task1_2.py`)** – Demand sizing, seasonality, uncertainty.
2. **Task 3 (`task3.py`)** – Assign ZIP3 → FC, compute demand shares.
3. **Task 4 (`task4.py`)** – Candidate clusters, feasibility, 90/10 reallocation.
4. **Task 5 (`task5.py`)** – Conversion, shipping, net revenue, GOP by OTD.
5. **Task 6 (`task6.py`)** – OTD optimization and extreme comparisons.
6. **Task 7 (`Task7.py`)** – Safety stock, rolling window targets.
7. **Task 8 (`Task8.py`)** – Chase vs Steady production strategy.

## Data Flow
- `data/` → consumed by Task 1–3.
- Task 1–2 → seasonality demand CSV → Task 5–8.
- Task 3 → assignment CSV → Task 5–7.
- Task 5 → financials → Task 6.
- Task 7 → daily FC targets → Task 8.
- All outputs → `output/{task}/`.

## How to Run
```bash
cd src
python task1_2.py
python task3.py
python task4.py
python task5.py
python task6.py
python Task7.py
python Task8.py
````

Each script creates subfolders in `output/`.
