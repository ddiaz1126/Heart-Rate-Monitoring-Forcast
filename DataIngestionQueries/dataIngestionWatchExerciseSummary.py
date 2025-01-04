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

watch_exercise_path = '/Users/daviddiaz/Desktop/Main/Heart-Rate-Monitoring-Forecast/GalaxyWatchData/com.samsung.health.exercise.2025010313.csv'

watch_exercise_df = pd.read_csv(watch_exercise_path, index_col=False)

filtered_watch_exercise_df = watch_exercise_df[['time_offset', 'update_time', 'end_time', 'pkg_name', 'duration', 'create_time', 'distance', 'count',
                                                'datauuid', 'max_heart_rate', 'max_speed', 'mean_speed', 'start_time', 'calorie', 'mean_heart_rate',
                                                'live_data', 'min_heart_rate', 'deviceuuid', 'exercise_type']]

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
    INSERT INTO watch_exercise_summary (
        time_offset, update_time, end_time, pkg_name, duration, create_time, distance, count,
        datauuid, max_heart_rate, max_speed, mean_speed, start_time, calorie, mean_heart_rate,
        live_data, min_heart_rate, deviceuuid, exercise_type
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    # Convert DataFrame to list of tuples
    rows_to_insert = filtered_watch_exercise_df.to_records(index=False).tolist()

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