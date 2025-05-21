# Nearest Large City Finder

## Overview

This Python utility enriches a CSV file containing cities and countries by adding a new column that identifies the nearest large city with a population above a specified threshold. It's useful for data analysis, geographic mapping, and understanding proximity relationships between locations.

![GeoCities Map](./images/map.jpeg)

## Features

- Maps each input city to its nearest large city based on geographical distance
- Uses a local dataset for fast lookups and geocoding
- Falls back to Google Maps or LocationIQ API for cities not found in the local dataset
- Handles errors gracefully and supports resuming interrupted processing
- Preserves existing results when processing specific row ranges
- Supports both Google Maps and LocationIQ geocoding APIs

## Requirements

- Python 3.6+
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - requests
  - python-dotenv

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain an API key for either Google Maps Geocoding API or LocationIQ API
4. Create a `.env` file in the project directory and add your API key:

```
GEOCODE_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

Run the script with default parameters:

```bash
python find_nearest_city.py --geocode_api_key YOUR_API_KEY
```

This will:
- Read from the default input file (`user_data/city_country.csv`)
- Find the nearest large city (population ≥ 1,000,000) for each input city
- Write results to the default output file (`user_data/city_country.nearest_cities.csv`)

### Input File Format

The input CSV file should contain at least two columns:
- `City`: The name of the city
- `Country`: The country where the city is located

Example:
```
City,Country
Vienna,Austria
Jersey City,United States
```

### Command-line Options

- `--input_file`: Path to the input CSV file (default: `user_data/city_country.csv`)
- `--output_file`: Path to the output CSV file (default: `user_data/city_country.nearest_cities.csv`)
- `--minimum_population_size`: Minimum population threshold for large cities (default: 1,000,000)
- `--geocode_api`: Which geocoding API to use ('googlemaps' or 'locationiq', default: 'googlemaps')
- `--geocode_api_key`: Your API key for the selected geocoding service
- `--starting_row`: First row to process (0-based, inclusive)
- `--ending_row`: Last row to process (0-based, inclusive)

### Examples

#### Using Google Maps API (default)

```bash
python find_nearest_city.py --geocode_api_key YOUR_GOOGLE_MAPS_API_KEY
```

#### Using LocationIQ API

```bash
python find_nearest_city.py --geocode_api locationiq --geocode_api_key YOUR_LOCATIONIQ_API_KEY
```

#### Specifying a Different Population Threshold

```bash
python find_nearest_city.py --geocode_api_key YOUR_API_KEY --minimum_population_size 500000
```

#### Using Custom Input and Output Files

```bash
python find_nearest_city.py --geocode_api_key YOUR_API_KEY --input_file my_cities.csv --output_file results.csv
```

#### Processing a Specific Range of Rows

```bash
python find_nearest_city.py --geocode_api_key YOUR_API_KEY --starting_row 5 --ending_row 10
```

This will process only rows 5 through 10 (inclusive, 0-based indexing) and preserve existing results for other rows.

### Using Environment Variables

Instead of passing the API key on the command line, you can store it in a `.env` file:

```
GEOCODE_API_KEY=your_api_key_here
```

## Output

The script adds a new column `Nearest Large City` to the input data. If no large city is found above the threshold, the cell will be blank.

Example output:
```
City,Country,Nearest Large City
Vienna,Austria,Vienna
Jersey City,United States,New York City
```

## Error Handling and Recovery

If the script encounters an error during processing (e.g., API rate limit exceeded), it will:
1. Log the error with details
2. Continue processing other rows if possible
3. Write partial results to the output file
4. Provide instructions on how to resume processing from the point of failure

To resume processing after an error:

```bash
python find_nearest_city.py --geocode_api_key YOUR_API_KEY --starting_row ROW_NUMBER
```

Where `ROW_NUMBER` is the row number indicated in the error message.

## Special Cases

- Cities with the value "n/a" are automatically assigned a blank nearest large city without performing any lookup
- NaN values in the city or country columns are handled gracefully
- The script preserves existing values in the `Nearest Large City` column when processing specific row ranges

## Performance Optimization

- The script caches API responses to reduce redundant API calls
- It logs the number of API calls made during processing
- At the end of processing, it reports the total number of API calls made
