# This file is to store the functions of this project
from datetime import datetime, timezone

def convert_time_to_seconds(time_str):
    """This function is to correctly transform the time metrics in strings cotaining colons in between into their respective unit"""
    try:
        # Split the time string into parts
        parts = list(map(int, time_str.split(':')))

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = parts
            return minutes + seconds / 60
        else:
            return None
    except Exception:
        return None
    
def convert_nanoseconds_to_datetime(timestamp):
    try:
        # Ensure the input is a valid number
        timestamp = int(timestamp)
        # Convert nanoseconds to seconds and create a naive datetime object
        return datetime.utcfromtimestamp(timestamp / 1e9)
    except (ValueError, TypeError):
        # Return None for invalid or null inputs
        return None
    
def convert_speed(elapsed_seconds, distance_meters):
    """Function to create speed based on elapsed seconds and distance in meters."""
    if elapsed_seconds is None or distance_meters is None:
        return None

    try:
        # Convert distance from meters to miles
        distance_miles = distance_meters * 0.000621371
    except (ValueError, TypeError):
        return None

    if elapsed_seconds <= 0:
        return None

    speed = distance_miles / (elapsed_seconds / 3600)  # convert seconds to hours
    return round(speed, 2)  # Output speed rounded to two decimal places

def run_query(query):
    try:
        from dotenv import load_dotenv
        import psycopg2
        import os
        import pandas as pd
        # Load environment variables
        load_dotenv()

        # Retrieve credentials from environment
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
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
        print(f"Now running query: {query}")
        print("...")
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        print("Data fetched and connection closed!")
        print(f"Dataframe Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_hist(col1):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Plot the figure
    plt.figure(figsize=(12,6))
    # Calculate the IQR
    q1 = np.percentile(col1, 25)
    q3 = np.percentile(col1, 75)
    iqr = q3 - q1

    # Calculate bin width using Freedman-Diaconis rule
    bin_width = 2 * iqr * len(col1) ** (-1 / 3)
    num_bins = int(np.ceil((col1.max() - col1.min()) / bin_width))

    counts, bins, _ = plt.hist(col1, bins=num_bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
    # Calculate the mean
    mean_value = col1.mean()

    # Add a red vertical line for the mean
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')

    plt.legend()
    plt.title(f"Histogram with Line Plot of {col1.name}")
    plt.xlabel(f"Values of {col1.name}")
    plt.ylabel("Frequency")
    plt.show()

def check_data(df):
    print("Missing Values:\n", df.isnull().sum())
    print("\nDuplicates:\n", df.duplicated().sum())

def visualize_scatter(col1, col2, regression=False, color='blue'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    plt.figure(figsize=(12,6))
    plt.scatter(col1, col2, color=color, label='Data Points')
    if regression == True:
        # Fit a linear regression model
        coefficients = np.polyfit(col1, col2, 1)
        trendline = np.polyval(coefficients, col1)
        plt.plot(col1, trendline, color='red', label="Trendline")
    plt.title(f"Scatter between {col1.name} & {col2.name}")
    plt.xlabel(f"{col1.name}")
    plt.ylabel(col2.name)
    plt.show()

def visualize_boxplot(*args):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # If arguments are passed as columns
    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        # Handle DataFrame
        df = args[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        data = [df[col].dropna() for col in numeric_cols]  # Drop NaN values for each column
        col_names = numeric_cols
    else:
        # Handle individual columns
        data = [col.dropna() for col in args]  # Drop NaN values for each column
        col_names = [col.name if hasattr(col, 'name') else f"Column {i+1}" for i, col in enumerate(args)]

    # Plotting the boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(data)
    
    # Set the title and labels
    plt.title("Boxplot of Variables")
    plt.xlabel("Variables")
    plt.ylabel("Values")
    plt.xticks(range(1, len(col_names) + 1), col_names)  # Set x-ticks to column names
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_lineplot(x, *args, **kwargs):
    """
    Create a line plot for the variables provided, with an option to scale y variables.

    Parameters:
    - x: The x-axis data (list, Series, or DataFrame column).
    - *args: One or more y variables (can be lists, Series, or DataFrame columns).
    - **kwargs: Additional keyword arguments for customization, such as:
        - scale (bool): Whether to scale y variables using Min-Max scaling. Default is False.
        - labels (list): Labels for the y variables.
        - title, xlabel, ylabel: Plot customization.
        - figsize: Size of the plot (default is (10, 6)).
        - linestyle: Style of the line.

    Returns:
    - A Matplotlib figure and axis objects.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert x to a Pandas Series if it isn't already
    if not isinstance(x, (pd.Series, list)):
        raise ValueError("x must be a list or Pandas Series")
    x = pd.Series(x)  # Ensure x is always a Series for consistency
    x = x.sort_values()

    # Initialize the plot
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

    # Plot each y variable
    for i, y in enumerate(args):
        label = kwargs.get("labels", [f"y{i+1}" for i in range(len(args))])[i]
        
        if isinstance(y, (pd.Series, list)):
            y_data = pd.Series(y)

            # Apply Min-Max scaling if requested
            if kwargs.get("scale", False):
                y_min, y_max = y_data.min(), y_data.max()
                y_data = (y_data - y_min) / (y_max - y_min)

            ax.plot(x, y_data, label=label, linestyle=kwargs.get("linestyle", '-'))

        elif isinstance(y, pd.DataFrame):
            # Handle numeric columns of a DataFrame
            for col in y.select_dtypes(include='number'):
                y_data = y[col]

                # Apply Min-Max scaling if requested
                if kwargs.get("scale", False):
                    y_min, y_max = y_data.min(), y_data.max()
                    y_data = (y_data - y_min) / (y_max - y_min)

                ax.plot(x, y_data, label=f"{col} ({label})", linestyle=kwargs.get("linestyle", '-'))

    # Customizations
    ax.set_xlabel(kwargs.get("xlabel", x.name))
    ax.set_ylabel(kwargs.get("ylabel", "Y-axis"))
    ax.set_title(kwargs.get("title", "Line Plot"))
    ax.legend(loc=kwargs.get("legend_loc", "best"))

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax

    