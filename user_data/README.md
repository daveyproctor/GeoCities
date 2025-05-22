# Experimental log for reproducibility

## Data quality problems manually solved:
- Charlottee,United States -> Charlotte
- Banglore,India -> Bangalore

## Run parameters:

```bash
. .venv/bin/activate
. .env
python find_nearest_city.py --max_cities_per_country 10 --minimum_population_size 500000
```
