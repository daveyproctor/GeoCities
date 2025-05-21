#!/usr/bin/env python3
"""
Find Nearest Large City Utility

This script enriches a user-provided CSV file by adding a new column 'nearest_large_city',
which maps each input city to the nearest large city with a population above a specified threshold.
"""

import argparse
import csv
import os
import urllib.parse
import requests
import math
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Cache for geocoding results to avoid redundant API calls
geocode_cache = {}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def load_simplemaps_dataset(file_path):
    """
    Load the SimpleMaps world cities dataset
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} cities from SimpleMaps dataset")
        return df
    except Exception as e:
        logger.error(f"Error loading SimpleMaps dataset: {e}")
        return pd.DataFrame()

def find_city_in_dataset(city, country, dataset):
    """
    Find a city in the SimpleMaps dataset
    """
    # Check for nan values
    if pd.isna(city) or pd.isna(country):
        logger.warning(f"Received nan value for city or country: city={city}, country={country}")
        return None
    
    # Try exact match first
    matches = dataset[(dataset['city_ascii'].str.lower() == city.lower()) & 
                      (dataset['country'].str.lower() == country.lower())]
    
    if not matches.empty:
        return matches.iloc[0]
    
    # Try fuzzy match on city name
    matches = dataset[(dataset['city_ascii'].str.lower().str.contains(city.lower())) & 
                      (dataset['country'].str.lower() == country.lower())]
    
    if not matches.empty:
        return matches.iloc[0]
    
    return None

def geocode_location_locationiq(city, country, api_key):
    """
    Geocode a city and country using LocationIQ API
    """
    # Check for nan values
    if pd.isna(city) or pd.isna(country):
        logger.warning(f"Received nan value for city or country: city={city}, country={country}")
        return (None, None)
    
    # Check cache first
    cache_key = f"{city.lower()},{country.lower()}"
    if cache_key in geocode_cache:
        return geocode_cache[cache_key]
    
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
            result = (lat, lon)
            
            # Cache the result
            geocode_cache[cache_key] = result
            return result
        else:
            logger.warning(f"No geocoding results found for {query}")
            return (None, None)
            
    except Exception as e:
        logger.error(f"Error geocoding {query}: {e}")
        return (None, None)

def find_nearest_large_city(lat, lon, dataset, min_population):
    """
    Find the nearest city with population >= min_population
    """
    if lat is None or lon is None:
        return None
    
    # Filter dataset for cities with population >= min_population
    large_cities = dataset[dataset['population'] >= min_population]
    
    if large_cities.empty:
        logger.warning(f"No cities found with population >= {min_population}")
        return None
    
    # Calculate distance to each large city
    distances = []
    for _, city in large_cities.iterrows():
        city_lat = city['lat']
        city_lon = city['lng']  # Note: SimpleMaps uses 'lng' not 'lon'
        distance = haversine_distance(lat, lon, city_lat, city_lon)
        distances.append((city['city'], distance))
    
    # Sort by distance and return the closest city
    distances.sort(key=lambda x: x[1])
    return distances[0][0] if distances else None

def process_input_file(input_file, output_file, simplemaps_dataset, min_population, api_key, starting_row=None, ending_row=None):
    """
    Process the input CSV file and add nearest_large_city column
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        simplemaps_dataset: DataFrame containing SimpleMaps dataset
        min_population: Minimum population threshold
        api_key: LocationIQ API key
        starting_row: First row to process (0-based, inclusive)
        ending_row: Last row to process (0-based, inclusive)
    """
    # Read input file
    input_df = pd.read_csv(input_file)
    total_rows = len(input_df)
    
    # Standardize column names to uppercase format
    column_mapping = {}
    for col in input_df.columns:
        if col.lower() == 'city':
            column_mapping[col] = 'City'
        elif col.lower() == 'country':
            column_mapping[col] = 'Country'
        elif col.lower() == 'nearest_large_city' or col.lower() == 'nearest large city':
            column_mapping[col] = 'Nearest Large City'
    
    # Rename columns if needed
    if column_mapping:
        input_df = input_df.rename(columns=column_mapping)
        logger.info(f"Standardized column names to uppercase format: {list(column_mapping.values())}")
    
    # Ensure required columns exist
    if 'City' not in input_df.columns or 'Country' not in input_df.columns:
        logger.error("Input file must contain 'City' and 'Country' columns (case-insensitive)")
        raise ValueError("Missing required columns in input file")
    
    # Check if output file already exists to preserve existing results
    output_df = None
    try:
        if os.path.exists(output_file):
            output_df = pd.read_csv(output_file)
            logger.info(f"Found existing output file with {len(output_df)} rows")
            
            # Standardize output column names too
            output_column_mapping = {}
            for col in output_df.columns:
                if col.lower() == 'city':
                    output_column_mapping[col] = 'City'
                elif col.lower() == 'country':
                    output_column_mapping[col] = 'Country'
                elif col.lower() == 'nearest_large_city' or col.lower() == 'nearest large city':
                    output_column_mapping[col] = 'Nearest Large City'
            
            if output_column_mapping:
                output_df = output_df.rename(columns=output_column_mapping)
            
            # Ensure output_df has the same structure as input_df
            if len(output_df) == len(input_df):
                # Keep the existing Nearest Large City column
                if 'Nearest Large City' in output_df.columns:
                    if 'Nearest Large City' not in input_df.columns:
                        input_df['Nearest Large City'] = output_df['Nearest Large City']
                    else:
                        # Preserve values but ensure we're working with the input structure
                        input_df['Nearest Large City'] = output_df['Nearest Large City']
                        logger.info("Preserving existing 'Nearest Large City' values from previous run")
            else:
                logger.warning("Existing output file has different number of rows than input file. Cannot preserve existing results.")
                output_df = None
    except Exception as e:
        logger.warning(f"Error reading existing output file: {e}. Will create a new output file.")
        output_df = None
    
    # Use input_df as our working dataframe
    df = input_df
    
    # Determine row range to process
    start = starting_row if starting_row is not None else 0
    end = ending_row if ending_row is not None else total_rows - 1
    
    # Ensure start and end are within valid range
    start = max(0, min(start, total_rows - 1))
    end = max(start, min(end, total_rows - 1))
    
    logger.info(f"Processing rows {start} to {end} (out of {total_rows} total rows)")
    
    # Create new column for nearest large city if it doesn't exist
    if 'Nearest Large City' not in df.columns:
        df['Nearest Large City'] = ''
    
    # Track the last successfully processed row
    last_successful_row = start - 1
    
    # Process each row in the specified range
    for index in range(start, end + 1):
        try:
            row = df.iloc[index]
            city = row['City']
            country = row['Country']
            
            logger.info(f"Processing {city}, {country}")
            
            # Special case for "n/a" cities
            if isinstance(city, str) and city.lower() == "n/a":
                logger.info(f"City is 'n/a', setting nearest_large_city to blank without lookup")
                df.at[index, 'Nearest Large City'] = ''
                last_successful_row = index
                continue
            
            # Try to find city in SimpleMaps dataset
            city_data = find_city_in_dataset(city, country, simplemaps_dataset)
            
            if city_data is not None:
                lat = city_data['lat']
                lon = city_data['lng']
                logger.info(f"Found {city}, {country} in SimpleMaps dataset: {lat}, {lon}")
            else:
                # Fallback to LocationIQ API
                logger.info(f"City not found in dataset, using LocationIQ API: {city}, {country}")
                lat, lon = geocode_location_locationiq(city, country, api_key)
            
            # Find nearest large city
            if lat is not None and lon is not None:
                nearest_city = find_nearest_large_city(lat, lon, simplemaps_dataset, min_population)
                df.at[index, 'Nearest Large City'] = nearest_city if nearest_city else ''
                logger.info(f"Nearest large city to {city}, {country}: {nearest_city}")
            else:
                logger.warning(f"Could not geocode {city}, {country}")
            
            # Update last successful row
            last_successful_row = index
            
        except Exception as e:
            # Log the error but continue processing
            logger.error(f"Error processing row {index} ({city}, {country}): {e}")
            logger.error(f"To resume processing from this point, use --starting_row={index}")
            # Continue with the next row
            continue
    
    try:
        # Write output file with all processed rows
        df.to_csv(output_file, index=False)
        logger.info(f"Output written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        logger.error(f"Last successfully processed row: {last_successful_row}. To resume, use --starting_row={last_successful_row + 1}")

def main():
    """
    Main function to parse arguments and run the script
    """
    parser = argparse.ArgumentParser(description='Find nearest large city for each city in input file')
    parser.add_argument('--input_file', default='user_data/city_country.csv',
                        help='Input CSV file with city and country columns')
    parser.add_argument('--output_file', default='user_data/city_country.nearest_cities.csv',
                        help='Output CSV file with nearest_large_city column added')
    parser.add_argument('--minimum_population_size', type=int, default=1000000,
                        help='Minimum population for a city to be considered large')
    parser.add_argument('--access_token', 
                        help='LocationIQ API token (can also be set in .env file as ACCESS_TOKEN)')
    parser.add_argument('--starting_row', type=int,
                        help='First row to process (0-based, inclusive)')
    parser.add_argument('--ending_row', type=int,
                        help='Last row to process (0-based, inclusive)')
    
    args = parser.parse_args()
    
    # Get API key from command line or environment variable
    api_key = args.access_token or os.getenv('ACCESS_TOKEN')
    if not api_key:
        logger.error("LocationIQ API token not provided. Use --access_token or set ACCESS_TOKEN in .env file")
        return
    
    # Load SimpleMaps dataset
    simplemaps_dataset = load_simplemaps_dataset('simplemaps/worldcities.csv')
    if simplemaps_dataset.empty:
        return
    
    try:
        # Process input file
        process_input_file(
            args.input_file,
            args.output_file,
            simplemaps_dataset,
            args.minimum_population_size,
            api_key,
            args.starting_row,
            args.ending_row
        )
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        logger.error("The output file may contain partial results up to the point of failure.")
        logger.error(f"To resume processing, use --starting_row parameter with the appropriate row number.")

if __name__ == '__main__':
    main()
