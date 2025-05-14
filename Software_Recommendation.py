
import json
import pandas as pd
with open('appraisals_dataset.json') as f:
    data = json.load(f)

print ("Data loaded successfully")
# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)
# Display the first few rows of the DataFrame
print(df.head())
# Display the columns of the DataFrame
print(df.columns)
# Display the shape of the DataFrame
print(df.shape)
# Display the data types of the columns
print(df.dtypes)
# Display the summary statistics of the DataFrame
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# Display the unique values in the 'appraisal' column
print(df['appraisal'].unique())
# Display the unique values in the 'software' column
print(df['software'].unique())
