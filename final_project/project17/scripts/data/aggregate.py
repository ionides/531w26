import pandas as pd
from pathlib import Path

# loading data
file_path = Path('polio_weekly_21-71.csv')
if not file_path.exists():
    raise FileNotFoundError(f"Could not find {file_path}")

raw = pd.read_csv(file_path, low_memory=False)

# convert to datetime objects
for col in ['PeriodStartDate', 'PeriodEndDate']:
    raw[col] = pd.to_datetime(raw[col], errors='coerce')

# cleaning and filtering data to get state-level weekly cases
state_cases = raw.loc[
    (raw['Fatalities'] == 0) &
    (raw['PartOfCumulativeCountSeries'] == 0) &
    (raw['Admin2Name'].isna()) &
    (raw['CityName'].isna())
].copy()

state_cases = state_cases.rename(columns={'CountValue': 'cases'})

# Create a 'year_month' column to group by month
state_cases['year_month'] = state_cases['PeriodStartDate'].dt.to_period('M')

# Group by month and sum cases
national_monthly_cases = state_cases.groupby('year_month')['cases'].sum().reset_index()

# Convert 'year_month' back to a timestamp for standard CSV date formatting
national_monthly_cases['date'] = national_monthly_cases['year_month'].dt.to_timestamp()
national_monthly_cases['year'] = national_monthly_cases['year_month'].dt.year

# filter exactly 5-year intervals starting from 1930 to 1964
start_year = 1930
end_year = start_year + 34
final_df = national_monthly_cases[
    (national_monthly_cases['year'] >= start_year) & 
    (national_monthly_cases['year'] <= end_year)
].copy()

# Sort, clean up columns, and prepare for export
final_df = final_df.sort_values('date')[['date', 'cases']]

# Export to CSV
output_filename = 'aggregated_monthly_polio_1930_1964.csv'
final_df.to_csv(output_filename, index=False)

print(f"Aggregation complete. Saved to {output_filename}")