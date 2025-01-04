# import libraries
from dotenv import load_dotenv
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

path = '/Users/daviddiaz/Desktop/Main/Heart-Rate-Monitoring-Forecast/GalaxyWatchData/com.samsung.shealth.tracker.pedometer_day_summary.2025010313.csv'
# Day Summary
watch_day_summary = pd.read_csv(path, delimiter=',', index_col=None)
day_summary_filtered = watch_day_summary[['recommendation', 'source_package_name', 'binning_data', 'create_time', 'deviceuuid', 'active_time', 'speed', 'achievement', 'healthy_step', 'datauuid', 'walk_step_count', 'run_step_count', 'day_time', 'calorie', 'distance', 'step_count', 'update_time']]


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
    INSERT INTO watch_summary (
        recommendation, source_package_name, binning_data, create_time, deviceuuid, 
        active_time, speed, achievement, healthy_step, datauuid, walk_step_count, 
        run_step_count, day_time, calorie, distance, step_count, update_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    # Convert DataFrame to list of tuples
    rows_to_insert = day_summary_filtered.to_records(index=False).tolist()

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