import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB (adjust the connection string as necessary)
client = MongoClient('localhost', 27017)
db = client['productDB']  

# Name of the collection you want to export
collection_name = 'products'  # Replace with your collection name
collection = db[collection_name]

# Fetch data from MongoDB
data = list(collection.find())

# Convert the MongoDB data to DataFrame
df = pd.DataFrame(data)

# Remove the '_id' column if you don't want to export MongoDB's internal IDs
if '_id' in df.columns:
    df.drop('_id', axis=1, inplace=True)

# Export to Excel
excel_file_path = 'mongodb_data.xlsx'  # Path where you want to save the Excel file
df.to_excel(excel_file_path, index=False)

print(f"Data exported to {excel_file_path}")
