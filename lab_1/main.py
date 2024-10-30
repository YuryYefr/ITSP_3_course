import pandas as pd
from helpers import float_to_time_format, print_results
from database import FlightsDatabase, compare_db_interaction

# Load the CSV file
csv_file = './flights.csv'
df = pd.read_csv(csv_file, delimiter=';')
column_names = ['year', 'month', 'day', 'flight_id', 'dep_city', 'dest_city', 'sched_dep_time', 'actual_dep_time',
                'dep_delay', 'sched_arr_time', 'act_arr_time', 'arr_delay']
df.columns = column_names
df = df.dropna()

# Apply the function to the 'hours' column
df['formatted_sched_dep_time'] = df['sched_dep_time'].apply(float_to_time_format)
df['formatted_act_dep_time'] = df['actual_dep_time'].apply(float_to_time_format)
df['formatted_sched_arr_time'] = df['sched_arr_time'].apply(float_to_time_format)
df['formatted_act_arr_time'] = df['act_arr_time'].apply(float_to_time_format)

# cleaning old records
df.drop(columns=['sched_dep_time', 'actual_dep_time', 'sched_arr_time', 'act_arr_time'], inplace=True)
df = df.sample(frac=0.1, random_state=1)

# populate databases
FlightsDatabase.populate_row_db(df)
FlightsDatabase.populate_col_db(df)

# compare requests
data = compare_db_interaction()
# Run the function to print results
print_results(data)
