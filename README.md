
## 🛠️ Prompt: Build a Python Utility to Map Cities to Nearest Large City

### Overview

Implement a Python CLI utility that enriches a user-provided Excel file by adding a new column `Nearest Large City`, which maps each input city to the nearest large city with a population above a specified threshold. The nearest city should be selected based on geographical distance (lat/lon), and the lookup must use a local dataset combined with the OpenCage geocoding API as a fallback.

### 🔧 Inputs

* **CSV file**: The user provides a CSV file (`user_data/city_country.csv`) with multiple columns, including:

  * `City` (e.g. "Vienna", "Jersey City")
  * `Country` (country name, e.g. "Austria" or "United States")

* **Command-line parameters**:

  * `--minimum_population_size`: Minimum population a nearby city must have to be considered "large enough" (e.g., 1000000)
  * `--access_token`: Your LocationIQ API token for geocoding (e.g., `--access_token=pk.32281ea63ece63588644f61d87889e0e`)

* **Dataset**: Use the [SimpleMaps World Cities](https://simplemaps.com/data/world-cities) CSV file, saved locally as simplemaps/worldcities.csv, with these columns:

  ```
  city, city_ascii, lat, lng, country, iso2, iso3, admin_name, capital, population, id
  ```

### 🔄 Behavior

1. **Geocode Input Cities**:

   * First, try to resolve the input city + country to lat/lon using the SimpleMaps dataset.
   * If no exact match is found locally, fallback to the [OpenCage Geocoding API](https://opencagedata.com/api) to get lat/lon.

2. **Find Nearest Large City**:

   * Use Haversine or other geographic distance metric to compute the closest city from the SimpleMaps dataset where:

     * `population >= minimum_population_size`
   * The "closest" city should be determined by actual geographic distance from the lat/lon of the input city.

3. **Write Output**:

   * Add a new column `nearest_large_city` to the user input CSV file.
   * Do not modify or delete any other columns in the user's input file.
   * Output should be `user_data/city_country.nearest_cities.csv`.


### 📋 Notes

* If no large city is found above the threshold, the `nearest_large_city` cell should be blank.
* Be efficient and cache API responses where applicable to reduce redundant API lookups.
* Handle ambiguous cities or missing geocodes gracefully with logging.

### 🌐 LocationIQ API Integration

As an alternative to OpenCage, you can use the LocationIQ API for geocoding. Here's how to integrate it:

#### API Overview
LocationIQ provides geocoding services to convert addresses or place names to geographic coordinates (latitude and longitude).

#### Authentication
- LocationIQ requires an API key (token) for authentication
- The key is passed as a query parameter `key` in the URL
- Example: `key=pk.32281ea63ece63588644f61d87889e0e`

#### Making Requests
- Base URL: `https://us1.locationiq.com/v1/search`
- Required parameters:
  - `key`: Your LocationIQ API token
  - `q`: The query string (city, country) that needs to be URL encoded
  - `format`: Response format (use `json`)

#### Example Request
```
https://us1.locationiq.com/v1/search?key=YOUR_API_KEY&q=Vienna%2C%20Austria&format=json
```

#### URL Encoding
- The query parameter (`q`) must be URL encoded
- Spaces become `%20`, commas become `%2C`, etc.
- In Python, you can use `urllib.parse.quote()` to encode the address string

#### Response Format
The API returns a JSON array of matching locations. Each location contains:
- `lat`: Latitude as a string (e.g., "51.5237629")
- `lon`: Longitude as a string (e.g., "-0.1584743")
- `display_name`: Full formatted address
- Additional metadata like `place_id`, `boundingbox`, etc.

#### Python Implementation Example
```python
import requests
import urllib.parse

def geocode_location(city, country, api_key):
    """
    Geocode a city and country using LocationIQ API.
    
    Args:
        city (str): City name
        country (str): Country name
        api_key (str): LocationIQ API key
        
    Returns:
        tuple: (latitude, longitude) as floats, or (None, None) if not found
    """
    # Construct and encode the query
    query = f"{city}, {country}"
    encoded_query = urllib.parse.quote(query)
    
    # Build the request URL
    url = f"https://us1.locationiq.com/v1/search?key={api_key}&q={encoded_query}&format=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        if data and len(data) > 0:
            # Extract latitude and longitude from the first result
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return (lat, lon)
        else:
            return (None, None)
            
    except Exception as e:
        print(f"Error geocoding {query}: {e}")
        return (None, None)
```

#### Error Handling
- Handle rate limiting (429 responses)
- Implement retries with exponential backoff
- Cache results to minimize API calls


### ✅ Deliverables

* A Python script or CLI utility that:

  * Accepts the input Excel filename and `--minimum_population_size`
  * Outputs an updated Excel file with `nearest_large_city` added

### 🚀 Running the Script

#### Prerequisites

Before running the script, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

#### Usage

The script can be run from the command line with the following syntax:

```bash
python find_nearest_city.py [options]
```

#### Command-line Options

* `--input_file`: Path to the input CSV file (default: `user_data/city_country.csv`)
* `--output_file`: Path to the output CSV file (default: `user_data/city_country.nearest_cities.csv`)
* `--minimum_population_size`: Minimum population threshold for large cities (default: 1000000)
* `--access_token`: Your LocationIQ API token (can also be set in .env file)
* `--starting_row`: First row to process (0-based, inclusive). If not provided, processing starts from the first row.
* `--ending_row`: Last row to process (0-based, inclusive). If not provided, processing continues to the last row.

#### Examples

Basic usage with default parameters:

```bash
python find_nearest_city.py
```

Specifying a different population threshold:

```bash
python find_nearest_city.py --minimum_population_size 500000
```

Specifying input and output files:

```bash
python find_nearest_city.py --input_file my_cities.csv --output_file results.csv
```

Providing the API token directly:

```bash
python find_nearest_city.py --access_token YOUR_LOCATIONIQ_API_KEY
```

Processing only a specific range of rows:

```bash
python find_nearest_city.py --starting_row 5 --ending_row 10
```

This will process only rows 5 through 10 (inclusive, 0-based indexing).

#### Using Environment Variables

Instead of passing the API token on the command line, you can store it in a `.env` file and export it to the shell env.

```
ACCESS_TOKEN=your_locationiq_api_key
```
