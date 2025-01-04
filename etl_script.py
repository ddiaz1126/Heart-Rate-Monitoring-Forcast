# import libraries
from dotenv import load_dotenv
from datetime import datetime, timezone
import psycopg2
import pandas as pd 
from psycopg2 import sql
import numpy as np
import os 

from functions import convert_nanoseconds_to_datetime
from functions import convert_speed


# Load environment variables
load_dotenv()

# Retrieve credentials from environment
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# Extract Data
try:
    # Connect to the database
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Connected to the database successfully!")

    # Create a cursor
    cursor = conn.cursor()

    # Queries for the three datasets
    map_my_run_query = """SELECT * FROM map_my_run_db;"""
    polar_data_query = """SELECT * FROM polar_data_db;"""
    watch_exercise_query = """SELECT * FROM watch_exercise_data_db;"""

    # Fetch data from map_my_run_db
    cursor.execute(map_my_run_query)
    map_my_run_data = cursor.fetchall()
    map_my_run_columns = [desc[0] for desc in cursor.description]
    map_my_run_df = pd.DataFrame(map_my_run_data, columns=map_my_run_columns)

    # Fetch data from polar_daba_db
    cursor.execute(polar_data_query)
    polar_data = cursor.fetchall()
    polar_data_columns = [desc[0] for desc in cursor.description]
    polar_data_df = pd.DataFrame(polar_data, columns=polar_data_columns)

    # Fetch data from watch_exercise_data_db
    cursor.execute(watch_exercise_query)
    watch_exercise_data = cursor.fetchall()
    watch_exercise_columns = [desc[0] for desc in cursor.description]
    watch_exercise_df = pd.DataFrame(watch_exercise_data, columns=watch_exercise_columns)

    # Close the cursor and connection
    cursor.close()
    conn.close()
    print("Data fetched and connection closed!")

except Exception as e:
    print(f"An error occurred: {e}")

print(polar_data_df.head(1))
map_my_run_df['datetime'] = map_my_run_df['time'].apply(convert_nanoseconds_to_datetime)

polar_data_df['datetime'] = polar_data_df['real_time'].apply(convert_nanoseconds_to_datetime)

watch_exercise_df['datetime'] = watch_exercise_df['start_time'].apply(convert_nanoseconds_to_datetime)

# Convert Speed (mi/hour)
map_my_run_df['speed'] = map_my_run_df.apply(lambda row: convert_speed(row['elapsed_seconds'], row['distance']), axis=1)

# Create elapsed_seconds
polar_data_df['first_datetime'] = polar_data_df.groupby('source_file')['datetime'].transform('first')
polar_data_df['elapsed_seconds'] = (polar_data_df['datetime'] - polar_data_df['first_datetime']).dt.total_seconds()

# Activity Id
polar_data_df = polar_data_df.rename(columns={'source_file': 'activity_id', 'hr': 'heart_rate', 'distances': 'distance'})
watch_exercise_df = watch_exercise_df.rename(columns={'live_data': 'activity_id'})

# Add nulls for missing fields
map_my_run_df['heart_rate'] = np.nan
polar_data_df['latitude'] = np.nan
polar_data_df['longitude'] = np.nan
polar_data_df['altitude'] = np.nan
watch_exercise_df['latitude'] = np.nan
watch_exercise_df['longitude'] = np.nan
watch_exercise_df['altitude'] = np.nan

# Application ID
watch_exercise_df['app_id'] = 0
polar_data_df['app_id'] = 1
map_my_run_df['app_id'] = 2

# Filter Columns
watch_exercise_df_filtered = watch_exercise_df[['activity_id', 'app_id', 'datetime', 'elapsed_seconds', 'heart_rate', 'speed', 'distance','latitude', 'longitude', 'altitude']]
polar_data_df_filtered = polar_data_df[['activity_id', 'app_id', 'datetime', 'elapsed_seconds', 'heart_rate', 'speed', 'distance','latitude', 'longitude', 'altitude']]
map_my_run_df_filtered = map_my_run_df[['activity_id', 'app_id', 'datetime', 'elapsed_seconds', 'heart_rate', 'speed', 'distance','latitude', 'longitude', 'altitude']]

# Concatenating the filtered DataFrames
combined_df = pd.concat(
    [watch_exercise_df_filtered, polar_data_df_filtered, map_my_run_df_filtered],
    axis=0,  # Concatenate along rows
    ignore_index=True  # Reindex the combined DataFrame
)

# Connect to the database
try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Connected to the database successfully!")

    # Create a cursor
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = sql.SQL("""
    INSERT INTO master_exercise_db (
        activity_id, app_id, datetime, elapsed_seconds, heart_rate, speed, distance, latitude, longitude, altitude
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    # Convert DataFrame to list of tuples
    rows_to_insert = combined_df.to_records(index=False).tolist()

    # Execute the batch insert
    cursor.executemany(insert_query, rows_to_insert)

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    print(f"{len(rows_to_insert)} rows inserted successfully.")
except Exception as e:
    print(f"An error occurred: {e}")