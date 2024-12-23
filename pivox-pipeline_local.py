import boto3
from datetime import datetime
import numpy as np

from pipeline_functions import *

# ground_tif = "C:/Users/RDCRLWKR/Documents/FileCloud/My Files/Active Projects/Snow Working Group/Pivox/Technical/Data/analysis pyvox/20240702-2000-46_bareGround.tif"

# Apply a filter to select files within a specific date range
start_date = datetime(2024, 10, 15) # installed in Boise on 1/25/2024
end_date = datetime(2024, 12, 31) #make sure this is beyond the window of interest
bare_ground_date = datetime(2024, 10, 7)

#Identify what files to process "new" or "all". new is all new files since last process, all is all files in the date range. All will rewrite .csv file
process_mode = "new"

# Set the S3 base bucket and the pivox folder
bucket_name = 'grid-dev-lidarscans'
pivox_name = 'Boise-Pivox/Freeman/'

# Initialize the S3 client
s3_client = boto3.client('s3')

# Procees the raw LAZ files from the S3 Bucket to ground points (.tif and .laz outputs)
las_filter_pipeline_leveled(bucket_name, pivox_name, s3_client, start_date, end_date, bare_ground_date, mode=process_mode)

# Calculate the snow depth for each scan using the bare ground file
sd_raster(bucket_name, pivox_name, s3_client, bare_ground_date, mode=process_mode)
snowdepth = snowdepth_timeseries(bucket_name, pivox_name, s3_client, mode=process_mode)

# scan_elev_df.to_csv('scan_snowdepth_df_WY2025.csv', index=False)
# # Save the flattened data dictionary as a .npy file
# np.save(r'C:/Users/RDCRLWKR/Documents/FileCloud/My Files/Active Projects/Snow Working Group/Pivox/Technical/Data/elevation_data_all_WY2025.npy', data_dict)