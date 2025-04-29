import pandas as pd
import numpy as np
import os
from plotnine import *
from scipy.stats import pearsonr


# ===================================================================================================================
# Reading in all data files
# ===================================================================================================================

def load_aligned_data_from_csv(directory="Modified Aligned Data All Participants"):
    """
    Loads saved CSV files and reconstructs the original dictionary structure:
    {
        'axial_rotation': {trial_num: df, ...},
        'flexion': {trial_num: df, ...},
        'lateral_bending': {trial_num: df, ...}
    }
    """
    aligned_data = {
        'axial_rotation': {},
        'flexion': {},
        'lateral_bending': {}
    }
    
    # Map filename prefixes to dictionary keys
    file_key_mapping = {
        'axial_rotation': 'axial_rotation',
        'flexion': 'flexion',
        'lateral': 'lateral_bending'  # Adjust if filenames differ
    }
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract motion type and trial number
            parts = filename.split('_')
            motion_type = parts[0]  # 'axial', 'flexion', 'lateral'
            trial_num = int(parts[-1].replace('.csv', '').split('_')[-1])
            
            # Find the correct dictionary key
            for key in file_key_mapping:
                if key in filename:
                    aligned_key = file_key_mapping[key]
                    break
            else:
                continue  # Skip unrecognized files
            
            # Load CSV and store in dictionary
            df = pd.read_csv(os.path.join(directory, filename))
            aligned_data[aligned_key][trial_num] = df
    
    return aligned_data

# Load all data
reconstructed_data = load_aligned_data_from_csv()

aligned_axial = reconstructed_data['axial_rotation']
aligned_flexion = reconstructed_data['flexion']
aligned_lateral = reconstructed_data['lateral_bending']

# ===================================================================================================================
# Graph All IMU or MoCap trials (by Motion)
# ===================================================================================================================

def plot_all_device_data(aligned_data, motion_name, device_type='IMU'):
    """
    Plot all trials for one device type using ggplot style
    
    Args:
        aligned_data: Dictionary of aligned DataFrames
        motion_name: Name of the motion
        device_type: 'IMU' or 'MoCap'
    """
    # Combine all trials into one DataFrame
    plot_data = []
    for trial_num, df in aligned_data.items():
        temp = df[['AlignedTime', f'Angle {device_type} (degrees)']].copy()
        temp['Trial'] = f'Trial {trial_num}'
        plot_data.append(temp)
    
    combined = pd.concat(plot_data)
    combined = combined.rename(columns={
        'AlignedTime': 'Time',
        f'Angle {device_type} (degrees)': 'Angle'
    })
    
    plot = (
        ggplot(combined, aes(x='Time', y='Angle', color='Trial')) +
        geom_line(size=1, alpha=0.7) +
        labs(
            title=f'{motion_name} - All {device_type} Trials',
            x='Aligned Time (s)',
            y='Angle (degrees)'
        ) +
        theme_bw() +
        theme(
            figure_size=(12, 6),
            legend_position='right',
            legend_title=element_blank()
        )
    )
    return plot

p = plot_all_device_data(aligned_lateral, 'Lateral', 'IMU')
# p.show()

p = plot_all_device_data(aligned_lateral, 'Lateral', 'MoCap')
# p.show()

# ===================================================================================================================
# Graph Differnce Plots (MoCap - IMU; by Motion)
# ===================================================================================================================

def plot_differences(aligned_data, motion_name, smooth=True):
    """
    Difference plot using ggplot-style with geom_smooth()
    Includes Pearson correlation coefficient in the title

    Args:
        aligned_data: Dictionary of aligned DataFrames
        motion_name: Name of the motion
        smooth: Add an overall smoothing spline
    """
    # Prepare difference data
    diff_data = []
    all_mocap = []
    all_imu = []

    for trial_num, df in aligned_data.items():
        temp = df[['AlignedTime']].copy()
        temp['Difference'] = df['Angle MoCap (degrees)'] - df['Angle IMU (degrees)']
        temp['Trial'] = f'Trial {trial_num}'
        diff_data.append(temp)

        # Collect all values for correlation calculation
        all_mocap.extend(df['Angle MoCap (degrees)'].values)
        all_imu.extend(df['Angle IMU (degrees)'].values)
    
    combined = pd.concat(diff_data)
    
    # Calculate Pearson correlation coefficient
    corr_coef, _ = pearsonr(all_mocap, all_imu)

    # Base plot
    plot = (
        ggplot(combined, aes(x='AlignedTime', y='Difference')) +
        geom_hline(yintercept=0, linetype='dashed', color='gray', alpha=0.5) +
        labs(
            title=f'{motion_name} - MoCap vs IMU Differences (r = {corr_coef:.3f})',
            x='Aligned Time (s)',
            y='Difference (degrees)'
        ) +
        theme_bw() +
        theme(figure_size=(12, 6))
    )
    
    if not smooth:
        # Individual trial lines with colors
        plot += geom_line(aes(color='Trial'), size=.85, alpha=0.5)
    else:
        # Faint individual lines
        plot += geom_line(aes(group='Trial'), color='gray', size=.85, alpha=0.15)
    
    if smooth:
        plot += geom_smooth(
            aes(group=1),  # Treat all data as one group
            method='loess',
            color='red',
            size=1.5,
            alpha=0.8,
            span=0.3,  # Controls smoothness (0-1; 1 = most smooth)
            se = False
        )
    
    return plot

p = plot_differences(aligned_axial, 'Axial Rotation', smooth=False)
# p.show()

p = plot_differences(aligned_axial, 'Axial Rotation', smooth=True)
# p.show()

p = plot_differences(aligned_flexion, 'Flexion', smooth=True)
# p.show()

p = plot_differences(aligned_lateral, 'Lateral', smooth=True)
# p.show()

# ===================================================================================================================
# Graph Individual Trial Plots (by Motion)
# ===================================================================================================================

def plot_single_trial(aligned_data, motion_name, trial_num, 
                                  gap_threshold=0.15,
                                  highlight_thresholds=None,
                                  highlight_color='red',
                                  highlight_alpha=0.1):
    if trial_num not in aligned_data:
        raise ValueError(f"Trial {trial_num} not found")
    
    df = aligned_data[trial_num].copy()
    
    # Add gap indicators
    df = df.sort_values('AlignedTime')
    df['time_diff'] = df['AlignedTime'].diff()
    df['new_group'] = (df['time_diff'] > gap_threshold).cumsum()
    
    # Base plot
    plot = (
        ggplot(df, aes(x='AlignedTime')) +
        geom_line(aes(y='Angle IMU (degrees)', color='"IMU"', group='new_group'), size=0.85) +
        geom_line(aes(y='Angle MoCap (degrees)', color='"MoCap"', group='new_group'), size=0.85) +
        labs(title=f'{motion_name} - Trial {trial_num}') +
        scale_color_manual(values=["blue", "green"]) +
        theme_bw() +
        theme(figure_size=(12, 6))
    )
    
    # Add threshold highlighting if requested
    if highlight_thresholds:
        lower, upper = highlight_thresholds
        
        # Create highlight regions
        highlight_df = pd.DataFrame({
            'xmin': df['AlignedTime'] - 0.005,
            'xmax': df['AlignedTime'] + 0.005,
            'ymin_upper': upper,
            'ymin_lower': -np.inf,
            'ymax_upper': np.inf,
            'ymax_lower': lower,
            'upper_highlight': (df['Angle IMU (degrees)'] > upper) | (df['Angle MoCap (degrees)'] > upper),
            'lower_highlight': (df['Angle IMU (degrees)'] < lower) | (df['Angle MoCap (degrees)'] < lower)
        })
        
        # Upper threshold highlights
        plot += geom_rect(
            data=highlight_df[highlight_df['upper_highlight']],
            mapping=aes(
                xmin='xmin',
                xmax='xmax',
                ymin='ymin_upper',
                ymax='ymax_upper'
            ),
            fill=highlight_color,
            alpha=highlight_alpha,
            inherit_aes=False
        )
        
        # Lower threshold highlights
        plot += geom_rect(
            data=highlight_df[highlight_df['lower_highlight']],
            mapping=aes(
                xmin='xmin', 
                xmax='xmax', 
                ymin='ymin_lower', 
                ymax='ymax_lower'
                ),
            fill=highlight_color,
            alpha=highlight_alpha,
            inherit_aes=False
        )
        
        # Threshold reference lines
        plot += geom_hline(yintercept=upper, linetype='dashed', color='black', alpha=0.5)
        plot += geom_hline(yintercept=lower, linetype='dashed', color='black', alpha=0.5)
        
    plot += labs(
        x='Aligned Time (s)',
        y='Difference (degrees)',
        color='Device'
        )
    
    return plot

p1 = plot_single_trial(aligned_axial, 'Axial Rotation', 25, highlight_thresholds=(-60, 60))
# p1.show()

p2 = plot_single_trial(aligned_flexion, 'Flexion', 25)
# p2.show()

p3 = plot_single_trial(aligned_lateral, 'Lateral', 25)
# p3.show()


p3 = plot_single_trial(aligned_axial, 'Lateral', 1, highlight_thresholds=(-60, 60))
p3.show()