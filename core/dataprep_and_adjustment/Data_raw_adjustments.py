import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroplan import FixedTarget

# ==========================================
# 1. The Wrapper Class
# ==========================================
class Target_ULTRASAT_Wrapped:
    def __init__(self, target_row):
        """A clean wrapper to hold real ULTRASAT targets and their metadata."""
        self.raw_data = target_row
        
        target_name = target_row['Target_ID']
        ra = target_row['RA (deg)']
        dec = target_row['Dec (deg)']
        
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        self.astroplan_target = FixedTarget(coord=coords, name=target_name)

# ==========================================
# 2. The Super-Table Builder
# ==========================================
def wrap_targets_and_times_for_astroplan(df):
    """
    Converts ALL columns containing 'MJD' into Astropy Time objects.
    Embeds Astroplan Target objects directly into the DataFrame.
    """
    print("Upgrading DataFrame to Astroplan-ready objects...")
    df_modified = df.copy()

    # --- A. Convert all MJD columns to Astropy Time Objects ---
    # Find every column header that contains the letters 'MJD'
    mjd_cols = [col for col in df_modified.columns if 'MJD' in str(col)]
    
    print(f"Wrapping {len(mjd_cols)} time columns into Time objects (this might take a few seconds)...")
    for col in mjd_cols:
        # We use a lambda function to check if the cell has a real number. 
        # If it does, wrap it in Time(). If it's empty (NaN), leave it as NaN.
        df_modified[col] = df_modified[col].apply(
            lambda x: Time(x, format='mjd') if pd.notna(x) else np.nan
        )

    # --- B. Create and Insert the Wrapper Objects ---
    wrapped_objects = []
    for index, row in df_modified.iterrows():
        # Pass the original row (before Time conversion) just in case you need raw numbers later
        wrapped_target = Target_ULTRASAT_Wrapped(df.iloc[index])
        wrapped_objects.append(wrapped_target)

    # Find exactly where 'Dec (deg)' is so we can insert right after it
    col_list = list(df_modified.columns)
    if 'Dec (deg)' in col_list:
        insert_loc = col_list.index('Dec (deg)') + 1 
        df_modified.insert(insert_loc, 'Astroplan_Target_Object', wrapped_objects)
    else:
        df_modified['Astroplan_Target_Object'] = wrapped_objects
        
    print("✅ Super-table successfully built!")
    return df_modified

def load_raw_data(file_path):
    """
    Reads the raw CSV file and returns it as a Pandas DataFrame.
    """
    print(f"Loading raw data from: {file_path}")
    raw_data_pd1 = pd.read_csv(file_path)
    return raw_data_pd1 


def add_peak_and_first_observation_data(df):
    """
    Scans the wide-format light curve data to find the brightest peak magnitudes, 
    the MJD times of those peaks, and the time of the very first observation.
    Inserts these new columns before the light curve arrays.
    """
    print("Calculating peak magnitudes and first observation times...")
    
    # We use a helper function to process row-by-row
    def calculate_row_stats(row):
        # Identify the relevant column groups
        time_cols = [c for c in row.index if str(c).startswith("Detection_Time_MJD_")]
        us_cols = [c for c in row.index if str(c).startswith("ULTRASAT_Magnitude_")]
        v_cols = [c for c in row.index if str(c).startswith("Visual_V_Magnitude_")]

        # 1. First Observation Time
        first_time = row[time_cols].min()

        # 2. ULTRASAT Peak (Remember: brighter = lower magnitude number)
        us_peak_mag = np.nan
        us_peak_time = np.nan
        if not row[us_cols].dropna().empty:
            us_min_col = row[us_cols].idxmin() # Finds the column name with the lowest value
            idx = us_min_col.split('_')[-1]    # Extracts the number (e.g., '108')
            us_peak_mag = row[us_min_col]
            us_peak_time = row[f"Detection_Time_MJD_{idx}"]

        # 3. Visual Peak
        v_peak_mag = np.nan
        v_peak_time = np.nan
        if not row[v_cols].dropna().empty:
            v_min_col = row[v_cols].idxmin()
            idx = v_min_col.split('_')[-1]
            v_peak_mag = row[v_min_col]
            v_peak_time = row[f"Detection_Time_MJD_{idx}"]

        return pd.Series([first_time, us_peak_mag, us_peak_time, v_peak_mag, v_peak_time])
    
    # Apply the math to the dataframe
    new_cols = [
        'First_Observation_Time_MJD', 
        'Peak_Mag_ULTRASAT', 'Peak_Time_MJD_ULTRASAT', 
        'Peak_Mag_Visual', 'Peak_Time_MJD_Visual'
    ]
    
    # We create a copy to avoid Pandas "SettingWithCopy" warnings
    df_modified = df.copy()
    df_modified[new_cols] = df_modified.apply(calculate_row_stats, axis=1)
    
    # ==========================================
    # REORDER COLUMNS
    # ==========================================
    # We want to pull these 5 new columns to the front, right before 'Detection_Time_MJD_1'
    all_cols = list(df_modified.columns)
    
    # Remove the new columns from the very end of the list
    for col in new_cols:
        all_cols.remove(col)
        
    # Find exactly where the massive light curve arrays begin
    insert_index = all_cols.index('Detection_Time_MJD_1')
    
    # Splice the list together: [Base Info] + [New Peak Info] + [Light Curve Arrays]
    final_col_order = all_cols[:insert_index] + new_cols + all_cols[insert_index:]
    
    return df_modified[final_col_order]