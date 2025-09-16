# Assumptions and Sources

## Demand

- Baseline demand (2025): 2,000,000 units.
- Growth scenarios (2026): +4% / +7.5% / +12%.
- Dobda 2025 share: 3.6%.  
- Share growth: +15%, +20%, +25%.

_Source: Casework instructions._

## Seasonality

- Factors from `demand_seasonalities.csv`.
- Random CVs: 20% (week), 15% (day), 15% (PMF).
- Uncertainty with ±1σ, ±2σ, ±2.5σ for 68/95/99%.

_Source: Case-provided seasonality file._

## Costs

- ASP: $3,000/unit.  
- COGS: $750/unit.  
- Shipping: distance- & OTD-based cost function.

_Source: Casework PDF._

## Service Levels

- Z-values: 68% = 1.0, 95% = 1.645, 99% = 2.33.
- Windows: 21-day (FC), 56-day (Network).

_Source: Standard z-tables, safety stock practices._

## Allocation

- 90% demand stays at preferred FC, 10% split to alternates.

## Production

- Chase strategy = meet daily target.  
- Steady strategy = fixed rate ensuring DC never < target.  
- Initial inventory = first-day target.
