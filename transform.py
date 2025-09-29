import pandas as pd
import time
from tqdm import tqdm  # Progress bar
from opencage.geocoder import OpenCageGeocode

# Load dataset
INPUT_FILE = "C:\\Users\\notclency\\Downloads\\Current_Year_Property_Assessments__Parcel__20250203.csv"
OUTPUT_FILE = "C:\\Users\\notclency\\Downloads\\"
API_KEY = "YOUR_OPENCAGE_API_KEY"  # Replace with your actual API key

# Initialize geocoder
geocoder = OpenCageGeocode(API_KEY)

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Check if 'Address' column exists
if "Address" not in df.columns:
    raise ValueError("The dataset must contain an 'Address' column.")

# Function to get latitude and longitude
def get_lat_lon(address):
    try:
        result = geocoder.geocode(address)
        if result:
            return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
    return None, None

# Apply function with progress bar
tqdm.pandas()
df[['Latitude', 'Longitude']] = df['Address'].progress_apply(lambda x: pd.Series(get_lat_lon(x)))

# Save updated dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"File saved as {OUTPUT_FILE}")