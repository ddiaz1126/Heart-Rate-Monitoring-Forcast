from dotenv import load_dotenv
from datetime import datetime
import psycopg2
import pandas as pd 
from psycopg2 import sql
import os 

# Load environment variables
load_dotenv()

# Retrieve credentials from environment
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

exercise_overview_path = '/Users/daviddiaz/Desktop/Main/Heart-Rate-Monitoring-Forecast/GalaxyWatchData/com.samsung.health.exercise.2025010313.csv'

exercise_overview = pd.read_csv(exercise_overview_path, index_col=False)

all_exercises = []  # List to hold each exercise DataFrame

for index, row in exercise_overview.iterrows():
    try:
        # Extract the name of the file
        file_name = row['live_data']
        # Attach File name to the path
        file_path = f"/Users/daviddiaz/Desktop/Main/Heart-Rate-Monitoring-Forecast/GalaxyWatchData/jsons/com.samsung.health.exercise/{file_name}.json"
        
        # Read the JSON file
        exercise_df = pd.read_json(file_path)
        
        # Create a new column to hold Minute Marker of the run
        exercise_df['elapsed_seconds'] = (exercise_df['start_time'] - exercise_df['start_time'].iloc[0]).dt.total_seconds().astype(int)

        # Create a Date Column
        exercise_df['Date'] = exercise_df['start_time'].dt.date.astype(str)

        # File Source
        exercise_df['live_data'] = row['live_data']

        # Append to the list
        all_exercises.append(exercise_df)
    except Exception as e:
    # Print an error message if there's an issue with reading the file
        print(f"Error reading file {file_path}: {e}")

concatenated_exercise_df = pd.concat(all_exercises, ignore_index=True)

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
    INSERT INTO watch_exercise_data_db (
        start_time, speed, heart_rate, elapsed_seconds, Date, live_data, cadence, distance
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """)

    # Convert DataFrame to list of tuples
    rows_to_insert = concatenated_exercise_df.to_records(index=False).tolist()

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