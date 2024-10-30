from mariadb import connect
from dotenv import load_dotenv
import os

load_dotenv()


# Function to convert float to HH:MM format
def float_to_time_format(float_hours):
    hours = int(float_hours) // 100  # Get the integer part for hours
    minutes = int(float_hours) % 100  # Get the remainder for minutes
    return f"{hours:02}:{minutes:02}"


def db_connector(port):
    connection = connect(
        host='localhost',
        port=port,
        user=os.getenv('USER'),
        password=os.getenv('PASSWORD'),
    )
    return connection


# Formatting function
def print_results(data):
    def format_time(time_obj):
        """Convert timedelta to hh:mm format."""
        hours, remainder = divmod(time_obj.seconds, 3600)
        minutes = remainder // 60
        return f"{hours:02}:{minutes:02}"

    print("Column-Based Results:")
    print("---------------------")
    print(
        f"- Average City Flights:\n    City: {data['col_avg_city_flights'][0]}, Average Flights: {data['col_avg_city_flights'][1]}")

    flight = data['col_flights_gt_average']
    print("- Flights Greater Than Average Delay:")
    print(
        f"    Date: {flight[0]}-{flight[1]:02}-{flight[2]:02}, Flight ID: {flight[3]}, From: {flight[4]}, To: {flight[5]}")
    print(
        f"    Scheduled Departure: {format_time(flight[6])}, Actual Departure: {format_time(flight[7])}, Departure Delay: {flight[8]} mins")
    print(
        f"    Scheduled Arrival: {format_time(flight[9])}, Actual Arrival: {format_time(flight[10])}, Arrival Delay: {flight[11]} mins")

    print(
        f"- Maximum City Delay:\n    City: {data['col_max_city_delay'][0]}, Delay: {data['col_max_city_delay'][1]} mins")
    print(
        f"- Minimum City Delay:\n    City: {data['col_min_city_delay'][0]}, Delay: {data['col_min_city_delay'][1]} mins")
    print(
        f"- Sum of City Delays:\n    City: {data['col_sum_city_delays'][0]}, Total Delay: {data['col_sum_city_delays'][1]} mins")

    print("\nRow-Based Results:")
    print("------------------")
    print(
        f"- Average City Flights:\n    City: {data['row_avg_city_flights'][0]}, Average Flights: {data['row_avg_city_flights'][1]}")

    flight = data['row_flights_gt_average']
    print("- Flights Greater Than Average Delay:")
    print(
        f"    Date: {flight[0]}-{flight[1]:02}-{flight[2]:02}, Flight ID: {flight[3]}, From: {flight[4]}, To: {flight[5]}")
    print(
        f"    Scheduled Departure: {format_time(flight[6])}, Actual Departure: {format_time(flight[7])}, Departure Delay: {flight[8]} mins")
    print(
        f"    Scheduled Arrival: {format_time(flight[9])}, Actual Arrival: {format_time(flight[10])}, Arrival Delay: {flight[11]} mins")

    print(
        f"- Maximum City Delay:\n    City: {data['row_max_city_delay'][0]}, Delay: {data['row_max_city_delay'][1]} mins")
    print(
        f"- Minimum City Delay:\n    City: {data['row_min_city_delay'][0]}, Delay: {data['row_min_city_delay'][1]} mins")
    print(
        f"- Sum of City Delays:\n    City: {data['row_sum_city_delays'][0]}, Total Delay: {data['row_sum_city_delays'][1]} mins")
