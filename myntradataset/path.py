import pandas as pd
import os

data = pd.read_csv('styles.csv')

# Create a new column named image_path
data['image_path'] = ''

# Update the image_path column with the path to the corresponding image
for index, row in data.iterrows():
    filename = f"{row['id']}.jpg"  # Assuming the filename is the id plus the file extension
    filepath = os.path.join('myntradataset\images', filename)
    data.at[index, 'image_path'] = filepath

# Save the updated CSV file
data.to_csv('styles3.csv', index=False)