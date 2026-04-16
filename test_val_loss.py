import pandas as pd
import numpy as np
df = pd.read_parquet('data/historical/final_processed/val_data.parquet')
print('Mean load:', df['load_mw'].mean())
print('Std load:', df['load_mw'].std())
