import os
from datetime import datetime
import pandas as pd
from pymongo import MongoClient

# Function to process each Excel file and update MongoDB
def process_excel_to_mongodb(excel_path, products_collection):
    data = pd.read_excel(excel_path)
    for box_id, group in data.groupby('BoxID'):
        side_id = datetime.now().strftime('%Y%m%d%H%M%S')
        texts = [{'Text': row['Text'], 'Area_per_character': row['Area_Ratio_Per_Character']} for _, row in group.iterrows()]
        
        existing_doc = products_collection.find_one({'BoxID': box_id, 'Sides.Texts': texts})
        if existing_doc:
            print(f"BoxID {box_id} with these texts already exists. Skipping update.")
            continue

        products_collection.update_one(
            {'BoxID': box_id}, 
            {
                '$push': {
                    'Sides': {
                        'SideID': side_id,
                        'Texts': texts
                    }
                }
            }, 
            upsert=True
        )

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['productDB']
products_collection = db['products']

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Loop over all Excel files in the directory
for file in os.listdir(script_dir):
    if file.endswith('.xlsx'):
        excel_path = os.path.join(script_dir, file)
        process_excel_to_mongodb(excel_path, products_collection)
        print(f"Processed {file}")

# Optionally, print the updated documents to verify the changes
for product in products_collection.find():
    print(product)
