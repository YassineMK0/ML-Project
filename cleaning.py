import pandas as pd

# Load the CSV file
df = pd.read_csv('output.csv')

print("Columns before dropping:", df.columns.tolist())

# Drop the 'name' column
df = df.drop(columns=['name'])

# Save back to CSV without the 'name' column
df.to_csv('output.csv', index=False)

# Reload to verify
df_check = pd.read_csv('output.csv')
print("Columns after dropping:", df_check.columns.tolist())
