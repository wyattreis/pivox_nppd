import pandas as pd
import numpy as np
import math
import gzip
import pdal
import json
import os
import laspy
import shutil
from os.path import join
from datetime import datetime
from pathlib import Path
from io import BytesIO
import rioxarray as rio
import rasterio
from rasterio.windows import from_bounds
import boto3
from botocore.exceptions import ClientError

# Function to extract date from the file path and convert it to a datetime object
def extract_date(file_path):
    date_str = file_path.split('/')[3].split('-')[0]
    return datetime.strptime(date_str, '%Y%m%d')

# Function to rotate the pivox point cloud based on the telemetry file
def rotation_matrix(s3_client,bucket_name, pivox_name, start_date, end_date):
    # Specify file paths
    prefix_telemetry =f'{pivox_name}telemetry/'
    t_files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_telemetry)
    
    # Specify the X and Y rotation columns in the CSVs
    columns_to_keep = [1, 4, 9, 12]

    # Create a list of all CSV files in the folder, skip the first since the station wasn't in Boise
    csv_files = [obj['Key'] for obj in t_files['Contents'] if obj['Key'].endswith('.gz')][1:]
    dataframes = []
    for csv in csv_files:
        obj = s3_client.get_object(Bucket=bucket_name, Key=csv)
        try:
            gzip_file_content = obj['Body'].read()
            with gzip.GzipFile(fileobj=BytesIO(gzip_file_content)) as gzipfile:
                try:
                    df = pd.read_csv(gzipfile, encoding='utf-8', header= None)
                    filtered_df = df.iloc[:, columns_to_keep]
                    dataframes.append(filtered_df)
                    # print(f"Processed {csv} with utf-8 encoding")
                except UnicodeDecodeError:
                    # If UTF-8 fails, try another encoding like ISO-8859-1
                    gzipfile.seek(0)  # Reset file pointer to the beginning
                    df = pd.read_csv(gzipfile, encoding='ISO-8859-1', header= None, on_bad_lines='skip')
                    filtered_df = df.iloc[:, columns_to_keep]
                    dataframes.append(filtered_df)
                    # print(f"Processed {csv} with ISO-8859-1 encoding")
        except Exception as e:
            print(f"Error processing file {csv}: {e}")

    new_column_names = ['date', 'time', 'X', 'Y']
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.columns = new_column_names[:combined_df.shape[1]]
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    filtered_df = combined_df[(combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)] # Date of Boise site install

    # Remove the outliers using the full period of interest
    def remove_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    clean_cols = ['X', 'Y']
    df_clean = remove_outliers(filtered_df, clean_cols)
    df_clean = df_clean.dropna()

    # Determine the mean rotation angles in the X and Y during the POI
    X_mean = df_clean['X'].mean()
    Y_mean = df_clean['Y'].mean()
    print(f'Mean Pivox rotation angles: X={X_mean}; Y={Y_mean}')

    # Convert the mean angles in degrees to radians for both the X and Y
    Xtheta_rad = math.radians(X_mean)
    Ytheta_rad = math.radians(Y_mean)

    # Calculate the cosine and sine of the angle and create the PDAL rotation Matrix for X and Y
    X_COStheta = math.cos(Xtheta_rad)
    X_SINtheta = math.sin(Xtheta_rad)
    Y_COStheta = math.cos(Ytheta_rad)
    Y_SINtheta = math.sin(Ytheta_rad)
    Xmatrix = f"1 0 0 0 0 {X_COStheta} {-X_SINtheta} 0 0 {X_SINtheta} {X_COStheta} 0 0 0 0 1"
    Ymatrix = f"{Y_COStheta} 0 {Y_SINtheta} 0 0 1 0 0 {-Y_SINtheta} 0 {Y_COStheta} 0 0 0 0 1"
    # print(f"X Matrix: {Xmatrix}")
    # print(f"Y Matrix: {Ymatrix}")
    return Xmatrix, Ymatrix

# Function to create and run a PDAL pipeline to extract the ground points
def las_filter_pipeline_dwnld(bucket_name, pivox_name, s3_client, start_date, end_date, Xmatrix, Ymatrix):

    # Set up sub directories within S3 bucket
    prefix_scans = f'{pivox_name}scans/' #levled-laz
    prefix_processed = f'{pivox_name}processed/'

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pivox_name, Delimiter='/')
    if 'CommonPrefixes' in response:
        print(f"Folder '{prefix_processed}' already exists in bucket '{bucket_name}'")
    else:
        s3_client.put_object(Bucket=bucket_name, Key=prefix_processed)
        print(f"Folder '{prefix_processed}' created in bucket '{bucket_name}'")

    # Identify all of the scans in the S3 Bucket
    scans = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_scans)
    scan_paths = [obj['Key'] for obj in scans.get('Contents', [])]
    scans_filtered = [file for file in scan_paths if start_date <= extract_date(file) <= end_date]
    print(f'Total Number of Files to Process: {len(scans_filtered)}')

    # Local directories for raw and results
    raw_dir = '/tmp/raw'
    results_dir = '/tmp/processed'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    #Xmatrix, Ymatrix = rotation_matrix(s3_client,bucket_name, pivox_name, start_date, end_date)

    for scan in scans_filtered:
        filename = os.path.basename(scan)
        local_file_path = None
        out_las = None
        if filename.endswith(('.laz', '.las')):
            print(f'Processing file: {filename}')
            basename = os.path.basename(scan)[:-4] # remove the .las from file path name
            local_file_path = os.path.join(raw_dir, basename)
            out_las = os.path.join(results_dir, f'{basename}_processed.laz')
            # Download .laz and .las files on S3 to local file path
            s3_client.download_file(bucket_name, scan, local_file_path)
            # print(f'Downloaded: {local_file_path}')
            
        elif filename.endswith('.gz'):
            print(f'Processing file: {filename}')
            basename = os.path.basename(scan)[:-7] # remove the .las.gz from file path name
            local_file_path = os.path.join(raw_dir, basename)
            out_las = os.path.join(results_dir, f'{basename}_processed.laz')
            out_tif = os.path.join(results_dir, f'{basename}_processed.tif')

            obj = s3_client.get_object(Bucket=bucket_name, Key=scan)
            with gzip.GzipFile(fileobj=obj['Body']) as gzipfile:
                with open(local_file_path, "wb") as f:
                    f.write(gzipfile.read())
            # print(f'Downloaded and unzipped: {local_file_path}')

        # PDAL Pipeline to process point clouds to just the "ground" returns
        pipeline = [
            {
                "type": "readers.las", 
                "filename": local_file_path,
                "override_srs": "EPSG:32611"
            },
            {
                "type": "filters.ferry",
                "dimensions": "=>tempY"
            },
            {
                "type": "filters.assign",
                "value": [
                    "tempY=Y",
                    "Y=Z",
                    "Z=tempY"
                    ]
            },
            {
                "type": "filters.transformation",
                "matrix": Xmatrix
            },
            {
                "type": "filters.transformation",
                "matrix": Ymatrix
            },
            {
                "type": "filters.crop",
                "bounds":"([0,40],[-30,20],[-5,5])" 
            },
            # {
            #     "type": "filters.assign",
            #     "assignment":"Classification[:]=0",
            #     "value": [
            #     "ReturnNumber = 1 WHERE ReturnNumber < 1",
            #     "NumberOfReturns = 1 WHERE NumberOfReturns < 1"
            #     ]
            # },
            # {
            #     "type": "filters.smrf",
            #     "cell": 0.5,
            #     "slope": 0.3,
            #     "threshold": 0.05,
            #     "window": 4
            # },
            # { 
            #     "type":"filters.range",
            #     "limits":"Classification[2:2]"
            # },
            {
                "type": "writers.las",
                "filename": out_las
            },
            {
                "type": "writers.gdal",
                "resolution": 0.01,
                "output_type": "mean",
                "filename": out_tif
            }
        ]

        pipeline_str = json.dumps(pipeline)
        pipeline_pdal = pdal.Pipeline(pipeline_str)
        
        try:
            pipeline_pdal.execute()
            # print(f"Processed file: {filename}")

            # Upload the results back to S3
            output_las_key = f'{prefix_processed}{os.path.basename(out_las)}'
            # output_tif_key = f'{prefix_processed}{os.path.basename(out_tif)}'
            
            s3_client.upload_file(out_las, bucket_name, output_las_key)
            # s3_client.upload_file(out_tif, bucket_name, output_tif_key)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
        finally:
            Path(local_file_path).unlink()
            Path(out_las).unlink(missing_ok=True)

# Function to create and run a PDAL pipeline to extract the ground points from the leveled point clouds 
def las_filter_pipeline_leveled(bucket_name, pivox_name, s3_client, start_date, end_date, bare_ground_date, mode="new"):
    """
    Function to process the leveled point clouds to extract ground points.
    
    Parameters:
    - bucket_name: str, S3 bucket name
    - pivox_name: str, PIVOX name prefix
    - s3_client: boto3 S3 client
    - start_date: datetime, start date for filtering
    - end_date: datetime, end date for filtering
    - bare_ground_date: datetime, specific date for bare ground data
    - mode: str, "new" to process only new files, "all" to process all files within date range
    """

    # Set up sub directories within S3 bucket
    prefix_scans = f'{pivox_name}scans/'
    prefix_processed = f'{pivox_name}processed/'

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pivox_name, Delimiter='/')
    existing_folders = [common_prefix['Prefix'] for common_prefix in response.get('CommonPrefixes', [])]

    if prefix_processed not in existing_folders:
        s3_client.put_object(Bucket=bucket_name, Key=prefix_processed)
        print(f"Folder '{prefix_processed}' created in bucket '{bucket_name}'")

     # List already processed files
    processed_files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_processed)
    processed_paths = {obj['Key']: True for obj in processed_files.get('Contents', []) if obj['Key'].endswith(('.las', '.tif'))}

    print(processed_paths)

    # Identify all of the scans in the S3 Bucket
    scans = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_scans)
    scan_paths = [obj['Key'] for obj in scans.get('Contents', []) if obj['Key'].endswith('.laz')]
    scan_leveled = [file for file in scan_paths if '.ROT' in file]

    print(scan_leveled)
    
    scans_filtered = []
    for file in scan_leveled:
        file_date = extract_date(file)
        basename = os.path.basename(file)
        out_las = f'{prefix_processed}{basename[:-4]}_processed.laz'
        out_tif = f'{prefix_processed}{basename[:-4]}_processed.tif'

        # limit number of files to process to new or all
        if mode == "new":
            if out_las not in processed_paths and out_tif not in processed_paths:
                if start_date <= file_date <= end_date or file_date == bare_ground_date:
                    scans_filtered.append(file)

        elif mode == "all":
            if start_date <= file_date <= end_date or file_date == bare_ground_date:
                    scans_filtered.append(file)
    
    print(scans_filtered)
    print(f'Total Number of New Files to Process: {len(scans_filtered)}')

    # Local directories for raw and results
    raw_dir = '/tmp/raw'
    results_dir = '/tmp/processed'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for scan in scans_filtered:
        filename = os.path.basename(scan)
        local_file_path = None
        out_las = None
        out_tif = None
        if filename.endswith(('.laz', '.las')):
            print(f'Processing file: {filename}')
            basename = os.path.basename(scan)[:-4] # remove the .las from file path name
            local_file_path = os.path.join(raw_dir, basename)
            out_las = os.path.join(results_dir, f'{basename}_processed.laz')
            out_tif = os.path.join(results_dir, f'{basename}_processed.tif')
            # Download .laz and .las files on S3 to local file path
            s3_client.download_file(bucket_name, scan, local_file_path)
            # print(f'Downloaded: {local_file_path}')
            
        elif filename.endswith('.gz'):
            # print(f'Processing file: {filename}')
            basename = os.path.basename(scan)[:-7] # remove the .las.gz from file path name
            local_file_path = os.path.join(raw_dir, basename)
            out_las = os.path.join(results_dir, f'{basename}_processed.laz')
            out_tif = os.path.join(results_dir, f'{basename}_processed.tif')

            obj = s3_client.get_object(Bucket=bucket_name, Key=scan)
            with gzip.GzipFile(fileobj=obj['Body']) as gzipfile:
                with open(local_file_path, "wb") as f:
                    f.write(gzipfile.read())
            # print(f'Downloaded and unzipped: {local_file_path}')

        # PDAL Pipeline to process point clouds to just the "ground" returns
        pipeline = [
            {
                "type": "readers.las", 
                "filename": local_file_path,
                "override_srs": "EPSG:32611"
            },
            # {
            #     "type": "filters.crop",
            #     "bounds":"([-1.5,0],[2,5],[-7,3])" 
            # },
            {
                "type": "filters.assign",
                "assignment":"Classification[:]=0",
                "value": [
                "ReturnNumber = 1 WHERE ReturnNumber < 1",
                "NumberOfReturns = 1 WHERE NumberOfReturns < 1"
                ]
            },
            {
                "type": "filters.smrf",
                "cell": 0.5,
                "slope": 0.3,
                "threshold": 0.05,
                "window": 4
            },
            { 
                "type":"filters.range",
                "limits":"Classification[2:2]"
            },
            {
                "type": "writers.las",
                "filename": out_las
            },
            {
                "type": "writers.gdal",
                "resolution": 0.01,
                "output_type": "mean",
                "filename": out_tif
            }
        ]

        pipeline_str = json.dumps(pipeline)
        pipeline_pdal = pdal.Pipeline(pipeline_str)
        
        try:
            pipeline_pdal.execute()
            # print(f"Processed file: {filename}")

            # Upload the results back to S3
            output_las_key = f'{prefix_processed}{os.path.basename(out_las)}'
            output_tif_key = f'{prefix_processed}{os.path.basename(out_tif)}'
            
            s3_client.upload_file(out_las, bucket_name, output_las_key)
            s3_client.upload_file(out_tif, bucket_name, output_tif_key)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
        # finally:
        #     # Path(local_file_path).unlink()
        #     Path(raw_dir).unlink(missing_ok=True)
        #     Path(results_dir).unlink(missing_ok=True)

    # delete the tmp file with all contents
    shutil.rmtree('/tmp/')


# Function to create snow depth rasters for each period using the data specified as the bare ground survey
def sd_raster(bucket_name, pivox_name, s3_client, bare_ground_date, mode='new'):
    """
    Processes snow depth rasters from processed TIFs by comparing with ground surface TIF. Upload TIF to S3 bucket.
    
    Parameters:
    - bucket_name: str, S3 bucket name.
    - pivox_name: str, PIVOX name prefix.
    - s3_client: boto3 S3 client.
    - bare_ground_date: datetime, date of the bare ground reference.
    - mode: str, 'all' to process all files or 'new' to process only new files.
    
    """
    # Set Pivox folder structure
    prefix_processed = f'{pivox_name}processed/'
    snowdepth_tif = f'{pivox_name}snowdepths/'

    # Create snowdepth directory if it doesnt exist already
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pivox_name, Delimiter='/')
    existing_folders = [common_prefix['Prefix'] for common_prefix in response.get('CommonPrefixes', [])]
    if snowdepth_tif not in existing_folders:
        s3_client.put_object(Bucket=bucket_name, Key=snowdepth_tif)
        print(f"Folder '{snowdepth_tif}' created in bucket '{bucket_name}'")

    # Identify all of the scans in the processed S3 Bucket
    processed = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_processed)
    processed_paths = [obj['Key'] for obj in processed.get('Contents', []) if obj['Key'].endswith('.tif')]

    # Identify all of the scans in the snowdepth S3 Bucket
    snowdepths = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=snowdepth_tif)
    sd_paths = [obj['Key'] for obj in snowdepths.get('Contents', []) if obj['Key'].endswith('.tif')]

    # Find the correct ground DEM file based on bare_ground_date
    bare_ground_filename = f"{bare_ground_date.strftime('%Y%m%d')}-"  # Match the start of the filename

    # Filter the processed paths to find the matching .ROT file for the bare ground date
    ground_tif = None
    for file in processed_paths:
        filename = os.path.basename(file)
        if filename.startswith(bare_ground_filename) and filename.endswith('.tif'):
            ground_tif = file
            processed_paths.remove(file)
            print(f"Matching ground TIFF found for bare ground date: {ground_tif}")
            break  # We found the matching file, so we can exit the loop

    if not ground_tif:
        print(f"No matching ground TIFF found for bare ground date.")
        return
  
    # Limit number of files to process to new or all
    new_snowdepth = []
    for file in processed_paths:
        basename = os.path.basename(file).split("_processed")[0].replace(".tif", "")
        sd_tif = f'{snowdepth_tif}{basename}_processed_sd.tif'

        # limit number of files to process to new or all
        if mode == "new":
            if sd_tif not in sd_paths:
                new_snowdepth.append(file)

        elif mode == "all":
            new_snowdepth.append(file)    
    
    # Allow for the function to exit if there are no new scans
    if len(new_snowdepth) == 0:
        print('No New Scans to Process - Exiting Raster Function')
        return
    else:
        print(f'Total Number of New Scans to Process: {len(new_snowdepth)}')

    # Local directories for processed files and snow depth files
    download_dir = '/tmp/processed'
    os.makedirs(download_dir, exist_ok=True)
    sd_dir = '/tmp/snowdepth'
    os.makedirs(sd_dir, exist_ok=True)

    # Download the ground DEM for bare_ground_date
    ground_tif_path = os.path.join(download_dir, os.path.basename(ground_tif))
    s3_client.download_file(bucket_name, ground_tif, ground_tif_path)
    print(f"Downloaded ground DEM for bare ground date as {os.path.basename(ground_tif)}")
    
    # Create DEMs of the snow depth using the ground raster
    data_dict = {} 
    
    for scan in new_snowdepth:
        filename = os.path.basename(scan)
        basename, ext = os.path.splitext(filename)
        surface_file_path = os.path.join(download_dir, filename)
        s3_client.download_file(bucket_name, scan, surface_file_path)

        # Extract the date and time from each file name
        date = basename[:8]
        time = basename[9:13]
        
        # Difference TIFF files and extract snow depth
        with rio.open_rasterio(ground_tif_path) as dem_ground, rio.open_rasterio(surface_file_path) as dem_surface:
            # Make sure the two DEMs are aligned
            if dem_ground.rio.crs != dem_surface.rio.crs:
                dem_surface = dem_surface.rio.reproject(dem_ground.rio.crs)
            
            dem_surface_resampled = dem_surface.rio.reproject_match(dem_ground)

            # Replace -9999 with NAN in both DEMs
            dem_ground = dem_ground.where(dem_ground != -9999, np.nan)
            dem_ground.rio.write_nodata(np.nan, inplace=True)
            dem_surface_resampled = dem_surface_resampled.where(dem_surface_resampled != -9999, np.nan)
            dem_surface_resampled.rio.write_nodata(np.nan, inplace=True)

            # Calculate Snow Depth
            dem_snowDepth = dem_surface_resampled - dem_ground

            # Flatten the snow depth array and store in data_dict
            elevation_data_flat = dem_snowDepth.values.flatten()  # Flatten the snow depth data
            data_dict[f'{date}_{time}'] = elevation_data_flat

            # Save snow depth to a temporary TIF file for uploading
            snow_depth_tif = os.path.join(sd_dir, f'{basename}_sd.tif')
            dem_snowDepth.rio.to_raster(snow_depth_tif)
            
        # Upload the snow depth TIFs back to AWS S3
        output_tif_key = f'{snowdepth_tif}{os.path.basename(snow_depth_tif)}'
        s3_client.upload_file(snow_depth_tif, bucket_name, output_tif_key)

        print(f"Snow Depth Calculated: {os.path.basename(snow_depth_tif)}")

        Path(surface_file_path).unlink()
        Path(snow_depth_tif).unlink()

    # # Upload the snow depth flittened dictionary to AWS S3
    # data_dict_local = f'{sd_dir}/snowdepth_dictionary.npy'
    # np.save(data_dict_local, data_dict)
    # output_tif_key = f'{snowdepth_tif}{os.path.basename(data_dict_local)}'
    # s3_client.upload_file(data_dict_local, bucket_name, output_tif_key)
    
    shutil.rmtree('/tmp/')
    return 

# Function to create a timeseries of ground heights from the sensor and standard deviation
def snowdepth_timeseries(bucket_name, pivox_name, s3_client, mode='new'):
    """
    Processes snow depth time series from processed TIFs by comparing with ground surface TIF.
    
    Parameters:
    - bucket_name: str, S3 bucket name.
    - pivox_name: str, PIVOX name prefix.
    - s3_client: boto3 S3 client.
    - bare_ground_date: datetime, date of the bare ground reference.
    - mode: str, 'all' to process all files or 'new' to process only new files.
    - existing_scan_key: str, S3 key of the existing scan_elev_df (optional).
    
    Returns:
    - scan_elev_df: pd.DataFrame, DataFrame containing mean and std deviation of snow depth for each scan.
    - data_dict: dict, flattened snow depth data for each timestamp.
    """
    
    # Set Pivox folder structure
    snowdepth_tif = f'{pivox_name}snowdepths/'
    csv_key = f'{snowdepth_tif}snowdepth.csv'
    npy_key = f'{snowdepth_tif}snowdepth_date.npy'

    # Local directories for raw and results
    sd_dir = '/tmp/snowdepth'
    os.makedirs(sd_dir, exist_ok=True)
    csv_path = '/tmp/snowdepth.csv'
    npy_path = '/tmp/snowdepth_date.npy'
    
    # Initialize or load existing CSV
    try:
        s3_client.head_object(Bucket=bucket_name, Key=csv_key)
        print(f"CSV file found at {csv_key}, downloading...")
        s3_client.download_file(bucket_name, csv_key, csv_path)
        existing_df = pd.read_csv(csv_path)

    except ClientError as e:
        # If file does not exist, catch the 404 error and return an empty DataFrame
        if e.response['Error']['Code'] == '404':
            print(f"No existing CSV found at {csv_key}, starting a new one.")
            existing_df = pd.DataFrame()
        else:
            # If any other error occurs, raise it
            print(f"Error occurred: {e}")
            raise

    # Initialize or load existing NPY
    try:
        s3_client.head_object(Bucket=bucket_name, Key=npy_key)
        print(f"NPY file found at {npy_key}, downloading...")
        s3_client.download_file(bucket_name, npy_key, npy_path)
        flat_cropped_data = np.load(npy_path, allow_pickle=True).item()

    except ClientError as e:
        # If file does not exist, catch the 404 error and return an empty DataFrame
        if e.response['Error']['Code'] == '404':
            print(f"No existing NPY found at {npy_key}, starting a new one.")
            flat_cropped_data = {}
        else:
            # If any other error occurs, raise it
            print(f"Error occurred: {e}")
            raise

    # Identify all of the scans in the snowdepth S3 Bucket
    snowdepths = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=snowdepth_tif)
    sd_paths = [obj['Key'] for obj in snowdepths.get('Contents', []) if obj['Key'].endswith('.tif')]

    new_snowdepth = [file for file in sd_paths if mode == 'all' or f'{os.path.basename(file).replace(".tif", "")}_processed_sd.tif' not in sd_paths]
    if not new_snowdepth:
        print('No new scans to process.')
        return existing_df

    print(f'Total new scans to process: {len(new_snowdepth)}')

    # Limit number of files to process to new or all
    # new_snowdepth = []
    # for file in sd_paths:
    #     basename = os.path.basename(file).split("_processed")[0].replace(".tif", "")
    #     sd_tif = f'{snowdepth_tif}{basename}_processed_sd.tif'

    #     # limit number of files to process to new or all
    #     if mode == "new" and sd_tif not in sd_paths:
    #         new_snowdepth.append(file)
    #     elif mode == "all":
    #         new_snowdepth.append(file)    
    
    # # Allow for the function to exit if there are no new scans
    # if not new_snowdepth:
    #     print('No new snow depths to add')
    #     return existing_df
    
    # print(f'Total number of new snow depths to add: {len(new_snowdepth)}')

    snowdepth = {}
    for scan in new_snowdepth:
        filename = os.path.basename(scan)
        basename, ext = os.path.splitext(filename)
        sd_file_path = os.path.join(sd_dir, filename)
        s3_client.download_file(bucket_name, scan, sd_file_path)

        date = basename[:8]
        time = basename[9:13]

        if (date, time) not in snowdepth:
            snowdepth[(date, time)] = {'date': date,'time': time}
        
        # Difference TIFF files and extract snow depth
        with rasterio.open(sd_file_path) as snowdepth_tiff:
            full_raster = snowdepth_tiff.read(1)
            full_raster = np.where(full_raster == snowdepth_tiff.nodata, np.nan, full_raster)

            # Calculate Snow Depth - full raster
            snowdepth[(date, time)]['sd_mean'] = np.nanmean(full_raster)
            snowdepth[(date, time)]['sd_std'] = np.nanstd(full_raster)

            # crop to area closest to sensor (min_x, min_y, max_x, max_y)
            aoi = (-5,1,5,6)
            window = from_bounds(*aoi, transform=snowdepth_tiff.transform)
            cropped_raster = snowdepth_tiff.read(1, window=window)
            cropped_raster = np.where(cropped_raster == snowdepth_tiff.nodata, np.nan, cropped_raster)

             # Flatten the cropped raster
            flat_cropped_data[f'{date}_{time}'] = cropped_raster.flatten()

            # Calculate Snow Depth - cropped area
            snowdepth[(date, time)]['sd_mean_cropped'] = np.nanmean(cropped_raster)
            snowdepth[(date, time)]['sd_std_cropped'] = np.nanstd(cropped_raster)
    
        print(f'Snow depth added from: {filename}')

    new_df = pd.DataFrame.from_dict(snowdepth, orient='index')
    updated_df = pd.concat([existing_df, new_df])
        
    # Re-upload the updated CSV to S3
    csv_path = '/tmp/updated_snowdepth.csv'
    updated_df.to_csv(csv_path, index=False)
    s3_client.upload_file(csv_path, bucket_name, csv_key)
    print(f'Updated CSV uploaded to {csv_key}.')

    # Update and save the NPY
    np.save(npy_path, flat_cropped_data)
    s3_client.upload_file(npy_path, bucket_name, npy_key)
    print(f"Updated NPY uploaded to {npy_key}.")

    # Cleanup
    shutil.rmtree(sd_dir)
    os.remove(csv_path)
    os.remove(npy_path)

    return updated_df