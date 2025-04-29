import pandas as pd
import glob
import os
import re
import numpy as np

# ===================================================================================================================
# Reading in all data files
# ===================================================================================================================

# Get all csv files in the folder
folder_path = "Aligned Data All Participants"

# Initialize separate dictionaries
axial_rotation = {}
flexion = {}
lateral_bending = {}

# Get all CSV files
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

for file in all_files:
    try:
        # Get the filename without path or extension
        filename = os.path.splitext(os.path.basename(file))[0]
        
        # Read the CSV file
        df = pd.read_csv(file)

        # Add source column (IMU or MoCap)
        if 'IMU' in filename.upper():
            df['Device'] = 'IMU'
        elif 'MOCAP' in filename.upper():
            df['Device'] = 'MoCap'
        else:
            df['Device'] = 'Unknown'
            print(f"Warning: Couldn't determine source for {filename}")
        
        # Add filename as a column for reference
        df['Filename'] = filename 
        
        # Categorize based on filename
        if 'AxialRotation' in filename:
            axial_rotation[filename] = df
        elif 'Flexion' in filename:
            flexion[filename] = df
        elif 'LateralBending' in filename:
            lateral_bending[filename] = df
        else:
            print(f"File {filename} didn't match any category")
            
    except Exception as e:
        print(f"Error loading {file}: {str(e)}")

print(f"Loaded {len(axial_rotation)} AxialRotation files")
print(f"Loaded {len(flexion)} Flexion files")
print(f"Loaded {len(lateral_bending)} LateralBending files")

# ===================================================================================================================
# Format each data file (extract trial number from file name)
# ===================================================================================================================

def parse_filename_components(df_dict):
    """Adds columns for trial number and time range extracted from filenames"""
    for name, df in df_dict.items():
        # Extract components using regular expression
        match = re.match(r'^(\d+)(.*?)_(IMU|MoCap)_(\d+s)_to_(\d+s)$', name)
        
        if match:
            # Add new columns
            df['TrialNumber'] = int(match.group(1))  # First number in filename
            df['TimeRange'] = f"{match.group(4)}-{match.group(5)}"  # Time range
            df['Duration'] = float(match.group(5).replace('s','')) - float(match.group(4).replace('s',''))
        else:
            print(f"Warning: Filename format not recognized for {name}")
            df['TrialNumber'] = None
            df['TimeRange'] = None
            df['Duration'] = None
    
    return df_dict

axial_rotation = parse_filename_components(axial_rotation)
flexion = parse_filename_components(flexion)
lateral_bending = parse_filename_components(lateral_bending)

# ===================================================================================================================
# Align times between IMU and MoCap data (round to nearest 10th of a second)
# ===================================================================================================================

def align_imu_mocap_data(imu_df, mocap_df):
    """
    Aligns IMU and MoCap data by:
    1. Rounding timestamps to the nearest 0.01s.
    2. Aggregating (mean) degree measures at each rounded time.
    3. Merging the two datasets on the rounded timestamps.
    4. Dropping rows with missing values.
    
    Args:
        imu_df: DataFrame containing IMU data.
        mocap_df: DataFrame containing MoCap data.
        
    Returns:
        merged_df: Aligned DataFrame with both IMU and MoCap data.
    """
    # Make copies to avoid modifying originals
    imu = imu_df.copy()
    mocap = mocap_df.copy()
    
    # Round time to nearest 0.01s
    imu['AlignedTime'] = imu['Shifted Time (s)'].round(2)
    mocap['AlignedTime'] = mocap['Time (s)'].round(2)
    
    # Aggregate (mean) degree measures at each rounded time
    imu_agg = imu.groupby('AlignedTime').agg({
        'Angle IMU (degrees)': 'mean',
        'TrialNumber': 'first',
        'TimeRange': 'first',
        'Duration': 'first',
        'Device': 'first',
        'Filename': 'first'
    }).reset_index()
    
    mocap_agg = mocap.groupby('AlignedTime').agg({
        'Angle MoCap (degrees)': 'mean',
        'TrialNumber': 'first',
        'TimeRange': 'first',
        'Duration': 'first',
        'Device': 'first',
        'Filename': 'first'
    }).reset_index()
    
    # Merge IMU and MoCap data on aligned time
    merged_df = pd.merge(
        imu_agg,
        mocap_agg,
        on=['AlignedTime', 'TrialNumber'],
        suffixes=('_IMU', '_MoCap')
    )
    
    # Drop rows where either IMU or MoCap data is missing
    merged_df = merged_df.dropna()
    
    merged_df = merged_df[['AlignedTime', 
                           'TrialNumber', 
                           'Angle IMU (degrees)', 
                           'Angle MoCap (degrees)', 
                           'Filename_IMU', 
                           'Filename_MoCap']]

    return merged_df

def process_motion(motion_dict, angle_col_name):
    """
    Processes a motion dictionary to align IMU/MoCap pairs with exact trial number matching.
    
    Args:
        motion_dict: Dictionary of DataFrames (axial_rotation, flexion, etc.)
        angle_col_name: Name of the angle column (e.g., 'AxialRotation Angle')
        
    Returns:
        Dictionary of aligned DataFrames keyed by trial number
    """
    aligned_data = {}
    
    # Create a lookup dictionary {trial_num: {device: df}}
    trial_lookup = {}
    for name, df in motion_dict.items():
        trial_num = df['TrialNumber'].iloc[0]
        device = 'IMU' if 'IMU' in name else 'MoCap'
        
        if trial_num not in trial_lookup:
            trial_lookup[trial_num] = {}
        trial_lookup[trial_num][device] = df
    
    # Process each trial
    for trial_num, devices in trial_lookup.items():
        try:
            if 'IMU' in devices and 'MoCap' in devices:
                # Standardize column names
                imu_df = devices['IMU'].rename(
                    columns={f'{angle_col_name} Angle IMU (degrees)': 'Angle IMU (degrees)'})
                mocap_df = devices['MoCap'].rename(
                    columns={f'{angle_col_name} Angle MoCap (degrees)': 'Angle MoCap (degrees)'})
                
                # Align data
                aligned_df = align_imu_mocap_data(imu_df, mocap_df)
            
                aligned_data[trial_num] = aligned_df
            else:
                print(f"Trial {trial_num} missing {'IMU' if 'IMU' not in devices else 'MoCap'} data")
                
        except Exception as e:
            print(f"Error processing trial {trial_num}: {str(e)}")
    
    return aligned_data

# Process all motion types
aligned_axial = process_motion(axial_rotation, 'AxialRotation')
aligned_flexion = process_motion(flexion, 'Flexion')
aligned_lateral = process_motion(lateral_bending, 'LateralBending')

# ===================================================================================================================
# Calculate Proportion of Time spent over THRESH Dregrees (THRESH = 60)
# ===================================================================================================================

# Threshold for considering an angle measurement "harmful"
THRESH = 60  

# Combine all movement data into a single DataFrame
frames = []
for movement, trial_dict in {
    'flexion': aligned_flexion,
    'lateral': aligned_lateral,
    'axial': aligned_axial
}.items():
    for participant_id, df in trial_dict.items():
        df_copy = df.copy()
        df_copy['participant'] = participant_id
        df_copy['movement'] = movement
        frames.append(df_copy)
        
all_data = pd.concat(frames, ignore_index=True)

# Reshape data to long format with device types as a column
long_format = all_data.melt(
    id_vars=['participant', 'movement', 'AlignedTime', 'TrialNumber'],
    value_vars=['Angle IMU (degrees)', 'Angle MoCap (degrees)'],
    var_name='device', 
    value_name='angle'
)

# Clean device names and flag harmful angles
long_format['device'] = long_format['device'].str.extract(r'(IMU|MoCap)')
long_format['harmful'] = long_format['angle'].abs() >= THRESH  # Using absolute value if negative angles matter

# Calculate harmful proportions by participant, movement and device
harmful_props = (
    long_format
    .groupby(['participant', 'movement', 'device'])['harmful']
    .agg(
        harmful_count='sum',         # Number of harmful measurements
        total_samples='size'       # Total number of measurements
    )
    .reset_index()
)

# Calculate proportion and rename columns
harmful_props['prop_harmful'] = harmful_props['harmful_count'] / harmful_props['total_samples']
result = harmful_props[['participant', 'movement', 'device', 'prop_harmful']]

# Sort for better readability
result = result.sort_values(['participant', 'movement', 'device'])


# ===================================================================================================================
# Save aligned data and Proportion in Bad Posture data
# ===================================================================================================================

# Create the directory if it doesn't exist
output_dir = "Modified Aligned Data All Participants"
os.makedirs(output_dir, exist_ok=True)

for trial_num, df in aligned_axial.items():
    df.to_csv(f"{output_dir}/axial_rotation_trial_{trial_num}.csv", index=False)

for trial_num, df in aligned_flexion.items():
    df.to_csv(f"{output_dir}/flexion_trial_{trial_num}.csv", index=False)

for trial_num, df in aligned_lateral.items():
    df.to_csv(f"{output_dir}/lateral_trial_{trial_num}.csv", index=False)


result.to_csv(f"prop_bad_posture_long.csv", index=False)
