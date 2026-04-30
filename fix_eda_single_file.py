import re

with open('src/eda.py', 'r') as f:
    code = f.read()

# 1. Update main data loading to use the combined dataset
old_load = """df = pd.read_csv("data/historical/raw/demand_load/load_sldc_2021-01-01_to_2026-04-06.csv")"""
new_load = """df = pd.read_csv("data/historical/raw/electricity_demand_2021-01-01_to_2026-04-06.csv")"""
code = code.replace(old_load, new_load)

# 2. Remove the weather loading and joining logic entirely, as it's now in the main df
weather_block_pattern = r"weather_df = pd\.read_csv\(\"data/historical/raw/weather/weather_openmeteo_2021-01-01_to_2026-04-06\.csv\"\).*?df_cpy = df_cpy\.join\(weather_df\[\[\"temperature\", \"humidity\"\]\], how=\"left\"\)"

code = re.sub(weather_block_pattern, "", code, flags=re.DOTALL)

with open('src/eda.py', 'w') as f:
    f.write(code)

