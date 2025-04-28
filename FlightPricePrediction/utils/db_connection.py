import pyodbc
import pandas as pd

def get_flight_data():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=sandesh;'
            'DATABASE=Flight;'
            'Trusted_Connection=yes;'
        )
        print("Connection to SQL Server established successfully.")
        query = """
        SELECT 
            Departure_Date, Origin, Destination, Economy, PremiumEconomy, Business, First,
            is_origin_holiday, is_destination_holiday, is_holiday_route
        FROM FlightPricePrediction
        WHERE Departure_Date IS NOT NULL AND Origin IS NOT NULL AND Destination IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f" Retrieved {len(df)} rows from the database.")
        return df
    except Exception as e:
        print("Failed to connect or fetch data from SQL Server.")
        print(f"Error: {e}")
        return pd.DataFrame()
