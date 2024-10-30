from helpers import db_connector

ROW_FLIGHT_DB_PORT = 3307
COL_FLIGHT_DB_PORT = 3308


class FlightsDatabase:
    """
    Controller for database interactions
    """

    @staticmethod
    def populate_row_db(df):
        connection = db_connector(ROW_FLIGHT_DB_PORT)
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS flights_row_db;")
        cursor.execute("USE flights_row_db;")

        cursor.execute("""
                CREATE TABLE IF NOT EXISTS flights_row (
                    year INT,
                    month INT,
                    day INT,
                    flight_id VARCHAR(10),
                    dep_city VARCHAR(100),
                    dest_city VARCHAR(100),
                    sched_dep_time TIME,
                    actual_dep_time TIME,
                    dep_delay DECIMAL(6,2),
                    sched_arr_time TIME,
                    act_arr_time TIME,
                    arr_delay DECIMAL(5,2)
                );
            """)
        # Insert data into MariaDB table
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO flights_row (year, month, day, flight_id, dep_city, dest_city, sched_dep_time, actual_dep_time, "
                "dep_delay, sched_arr_time, act_arr_time, arr_delay) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (row['year'], row['month'], row['day'], row['flight_id'], row['dep_city'], row['dest_city'],
                 row['formatted_sched_dep_time'], row['formatted_act_dep_time'], row[
                     'dep_delay'], row['formatted_sched_arr_time'], row['formatted_act_arr_time'], row['arr_delay'])
            )
        connection.commit()
        cursor.close()
        connection.close()
        print("ROW Database and tables created successfully.")

    @staticmethod
    def get_sum_city_delays(cursor, flight_data_table):
        sql = """SELECT dep_city, 
        SUM(dep_delay) AS total_dep_delay
            FROM {}
            GROUP BY dep_city""".format(flight_data_table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def calculate_avg_city_flights(cursor, flight_data_table):
        sql = """
                SELECT dep_city, COUNT(*) AS flight_count
                FROM {}
                GROUP BY dep_city;
                """.format(flight_data_table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def get_min_city_delay(cursor, flight_data_table):
        sql = """
            SELECT dep_city, SUM(dep_delay) AS total_dep_delay
            FROM {}
            GROUP BY dep_city
            ORDER BY total_dep_delay
            LIMIT 1;
        """.format(flight_data_table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def get_max_city_delay(cursor, flight_data_table):
        sql = """
                SELECT dep_city, SUM(dep_delay) AS total_dep_delay
                FROM {}
                GROUP BY dep_city
                ORDER BY total_dep_delay DESC
                LIMIT 1;
                """.format(flight_data_table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def get_flights_gt_average_delay(cursor, flight_data_table):
        sql = """
            SELECT *
            FROM {}
            WHERE dep_delay > (SELECT AVG(dep_delay) FROM {});
        """.format(flight_data_table, flight_data_table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def populate_col_db(df):
        connection = db_connector(COL_FLIGHT_DB_PORT)
        cursor = connection.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS flights_column_db;")
        cursor.execute("USE flights_column_db;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flights_column (
                year INT ,
                month INT ,
                day INT ,
                flight_id VARCHAR(10) ,
                dep_city VARCHAR(100) ,
                dest_city VARCHAR(100) ,
                sched_dep_time TIME ,
                actual_dep_time TIME ,
                dep_delay DECIMAL(6,2) ,
                sched_arr_time TIME ,
                act_arr_time TIME ,
                arr_delay DECIMAL(5,2)
            );
        """)
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO flights_column (year, month, day, flight_id, dep_city, dest_city, sched_dep_time, actual_dep_time, "
                "dep_delay, sched_arr_time, act_arr_time, arr_delay) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (row['year'], row['month'], row['day'], row['flight_id'], row['dep_city'], row['dest_city'],
                 row['formatted_sched_dep_time'], row['formatted_act_dep_time'], row[
                     'dep_delay'], row['formatted_sched_arr_time'], row['formatted_act_arr_time'], row['arr_delay'])
            )

        connection.commit()
        cursor.close()
        connection.close()
        print("COL Database and tables created successfully.")

def compare_db_interaction():
    result = {}
    #ROW section
    row_db_connection = db_connector(ROW_FLIGHT_DB_PORT)
    row_data_table = 'flights_row'
    cursor = row_db_connection.cursor()
    cursor.execute("USE flights_row_db;")
    row_sum_city_delays = FlightsDatabase.get_sum_city_delays(cursor, row_data_table)
    row_avg_city_flights = FlightsDatabase.calculate_avg_city_flights(cursor, row_data_table)
    row_max_city_delay = FlightsDatabase.get_max_city_delay(cursor, row_data_table)
    row_min_city_delay = FlightsDatabase.get_min_city_delay(cursor, row_data_table)
    row_flights_gt_average = FlightsDatabase.get_flights_gt_average_delay(cursor, row_data_table)
    cursor.close()
    row_db_connection.close()
    result.update({'row_sum_city_delays': row_sum_city_delays})
    result.update({'row_avg_city_flights': row_avg_city_flights})
    result.update({'row_max_city_delay': row_max_city_delay})
    result.update({'row_min_city_delay': row_min_city_delay})
    result.update({'row_flights_gt_average': row_flights_gt_average})

    # COL section
    col_db_connection = db_connector(COL_FLIGHT_DB_PORT)
    col_data_table = 'flights_column'
    cursor = col_db_connection.cursor()
    cursor.execute("USE flights_column_db;")
    col_sum_city_delays = FlightsDatabase.get_sum_city_delays(cursor, col_data_table)
    col_avg_city_flights = FlightsDatabase.calculate_avg_city_flights(cursor, col_data_table)
    col_max_city_delay = FlightsDatabase.get_max_city_delay(cursor, col_data_table)
    col_min_city_delay = FlightsDatabase.get_min_city_delay(cursor, col_data_table)
    col_flights_gt_average = FlightsDatabase.get_flights_gt_average_delay(cursor, col_data_table)
    cursor.close()
    col_db_connection.close()
    result.update({'col_sum_city_delays': col_sum_city_delays})
    result.update({'col_avg_city_flights': col_avg_city_flights})
    result.update({'col_max_city_delay': col_max_city_delay})
    result.update({'col_min_city_delay': col_min_city_delay})
    result.update({'col_flights_gt_average': col_flights_gt_average})
    return result
