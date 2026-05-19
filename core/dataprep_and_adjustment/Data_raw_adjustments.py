import pandas as pd
import numpy as np
import math
from IPython.display import display
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroplan import FixedTarget
import matplotlib.pyplot as plt



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
    
    def calculate_row_stats(row):
        #This function calculates the first observation time, the peak magnitude and its time for both ULTRASAT and Visual bands for a single row of the DataFrame.
        # Identify the relevant column groups
        time_cols = [c for c in row.index if str(c).startswith("Detection_Time_MJD_")] #list of all the columns that start with "Detection_Time_MJD_"
        us_cols = [c for c in row.index if str(c).startswith("ULTRASAT_Magnitude_")] #list of all the columns that start with "ULTRASAT_Magnitude_"
        v_cols = [c for c in row.index if str(c).startswith("Visual_V_Magnitude_")] #list of all the columns that start with "Visual_V_Magnitude_"

        # 1. First Observation Time
        first_time = row[time_cols].min()

        # 2. ULTRASAT Peak (brighter = lower magnitude number)
        us_peak_mag = np.nan
        us_peak_time = np.nan
        if not row[us_cols].dropna().empty:
            us_min_col = row[us_cols].idxmin() # Finds the column name with the lowest value
            idx = us_min_col.split('_')[-1]    # splits the column name to extract the index number at the end
            us_peak_mag = row[us_min_col] # Gets the actual magnitude value from that column
            us_peak_time = row[f"Detection_Time_MJD_{idx}"] # Uses the same index to find the corresponding time from the time columns

        # 3. Visual Peak
        v_peak_mag = np.nan
        v_peak_time = np.nan
        if not row[v_cols].dropna().empty:
            v_min_col = row[v_cols].idxmin()
            idx = v_min_col.split('_')[-1]
            v_peak_mag = row[v_min_col]
            v_peak_time = row[f"Detection_Time_MJD_{idx}"]

        return pd.Series([first_time, us_peak_mag, us_peak_time, v_peak_mag, v_peak_time])
    
    
    new_cols = [
        'First_Observation_Time_MJD', 
        'Peak_Mag_ULTRASAT', 'Peak_Time_MJD_ULTRASAT', 
        'Peak_Mag_Visual', 'Peak_Time_MJD_Visual'
    ]
    
    # We create a copy to avoid Pandas "SettingWithCopy" warnings
    df_modified = df.copy() #create a copy of the original DataFrame to avoid modifying it directly
    df_modified[new_cols] = df_modified.apply(calculate_row_stats, axis=1) #apply the calculate_row_stats function to each row of the DataFrame and assign the resulting Series to the new columns
    
    # We want to pull these 5 new columns to the front, right before 'Detection_Time_MJD_1'
    all_cols = list(df_modified.columns) # Get the current list of all column names in the modified DataFrame as a list. This will be used to rearrange the columns in the desired order.
    
    # Remove the new columns from the very end of the list
    for col in new_cols:
        all_cols.remove(col)
        
    # Find exactly where the massive light curve arrays begin
    insert_index = all_cols.index('Detection_Time_MJD_1')
    
    # Splice the list together: [Base Info] + [New Peak Info] + [Light Curve Arrays]
    final_col_order = all_cols[:insert_index] + new_cols + all_cols[insert_index:]
    
    return df_modified[final_col_order] #reorder the DataFrame columns according to the new_col_order list and return the modified DataFrame 



class LightCurvePlotting:
    def __init__(self, raw_data_df):
        self.data_df = raw_data_df

    def _extract_target_row(self, target_identifier):
        if isinstance(target_identifier, (pd.Series, dict)):
            return target_identifier
        elif isinstance(target_identifier, str):
            row = self.data_df[self.data_df['Target_ID'] == target_identifier]
            if row.empty:
                raise ValueError(f"Target ID '{target_identifier}' not found in the data.")
            return row.iloc[0]
        raise TypeError("Expected a Target_ID string or a DataFrame row (Pandas Series).")

    def _extract_band_data(self, target_row, band_prefix="ULTRASAT"):
        times, mags = [], []
        mag_prefix = f"{band_prefix}_Magnitude_" if band_prefix == "ULTRASAT" else "Visual_V_Magnitude_"
        
        # Iterate over time columns and match them directly to magnitude columns
        for col in target_row.index[target_row.index.str.startswith("Detection_Time_MJD_")]:
            idx = col.split('_')[-1]
            mag_col = f"{mag_prefix}{idx}"
            
            t, m = target_row.get(col, np.nan), target_row.get(mag_col, np.nan)
            if pd.notna(t) and pd.notna(m):
                times.append(t)
                mags.append(m)
                
        # Sort by time to ensure lines plot correctly
        if times:
            times, mags = zip(*sorted(zip(times, mags)))
            
        return np.array(times), np.array(mags)

    def _convert_time(self, time_data, time_format, peak_time=None):
        if time_format == "UTC":
            return Time(time_data, format='mjd').datetime, time_format
            
        elif time_format in ["relative_days", "relative_hours"]:
            if pd.isna(peak_time):
                print("Warning: Missing peak time. Defaulting to MJD.")
                return time_data, "MJD"
                
            rel_time = time_data - peak_time
            if time_format == "relative_hours":
                rel_time *= 24
            return rel_time, time_format
            
        return time_data, time_format

    def _add_event_line(self, ax, time_val, color, linestyle, label, time_format, peak_time=None):
        if pd.isna(time_val): return
        time_val, _ = self._convert_time(time_val, time_format, peak_time)
        ax.axvline(x=time_val, color=color, linestyle=linestyle, label=label, alpha=0.7, linewidth=1.5)

    def _finalize_plot(self, ax, target_row, band_name, time_format="MJD"):
        target_id = target_row['Target_ID']
        
        def mjd_to_date_str(mjd_val):
            return Time(mjd_val, format='mjd').datetime.strftime("%d/%m/%Y")

        info_parts = []
        events = {
            'First Obs': 'First_Observation_Time_MJD',
            'US Peak': 'Peak_Time_MJD_ULTRASAT',
            'Vis Peak': 'Peak_Time_MJD_Visual'
        }
        
        for label, col_name in events.items():
            val = target_row.get(col_name, np.nan)
            if pd.notna(val) and (label == 'First Obs' or (label == 'US Peak' and "ULTRASAT" in band_name) or (label == 'Vis Peak' and "Visual" in band_name) or "Multiband" in band_name):
                info_parts.append(f"{label}: {mjd_to_date_str(val)}")

        main_title = f"Light Curve for {target_id} ({band_name})"
        ax.set_title(f"{main_title}\n{' | '.join(info_parts)}" if info_parts else main_title, fontsize=10)
        
        labels = {
            "UTC": "Time (UTC)",
            "relative_days": "Time since Peak (Days)",
            "relative_hours": "Time since Peak (Hours)",
            "MJD": "Time (MJD)"
        }
        ax.set_xlabel(labels.get(time_format, "Time (MJD)"))
        if time_format == "UTC": ax.tick_params(axis='x', rotation=45)
            
        ax.set_ylabel("Magnitude (Apparent)")
        ax.invert_yaxis() 
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    def plot_single_band(self, target_identifier, band="ULTRASAT", time_format="MJD"):
        target_row = self._extract_target_row(target_identifier)
        times, mags = self._extract_band_data(target_row, band_prefix=band)
        
        color = 'purple' if band == "ULTRASAT" else 'green'
        peak_col = f'Peak_Time_MJD_{band}'
        peak_time = target_row.get(peak_col, np.nan)
        
        times, final_format = self._convert_time(times, time_format, peak_time)
        
        _, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(times, mags, color=color, marker='o', s=20, label=f'{band} Data')
        ax.plot(times, mags, color=color, linestyle='-', alpha=0.4)
        
        self._add_event_line(ax, target_row.get('First_Observation_Time_MJD'), 'black', '--', 'First Obs', final_format, peak_time)
        self._add_event_line(ax, peak_time, color, ':', f'{band} Peak', final_format, peak_time)
            
        self._finalize_plot(ax, target_row, f"{band} Band", final_format) # FIX APPLIED HERE
        plt.tight_layout()
        plt.show()

    def plot_multiband_lc(self, target_identifier, time_format="MJD"):
        target_row = self._extract_target_row(target_identifier)
        
        if time_format in ["relative_days", "relative_hours"]:
            print("Note: Relative time plotting is disabled for multiband curves. Defaulting to MJD.")
            time_format = "MJD"
            
        _, ax = plt.subplots(figsize=(10, 6))
        
        bands = [("ULTRASAT", 'purple', 'o', 'Peak_Time_MJD_ULTRASAT'), 
                 ("Visual", 'green', 'X', 'Peak_Time_MJD_Visual')]
                 
        for band, color, marker, peak_col in bands:
            t, m = self._extract_band_data(target_row, band_prefix=band)
            t, final_format = self._convert_time(t, time_format)
            ax.scatter(t, m, color=color, marker=marker, s=25, label=f'{band} Data')
            ax.plot(t, m, color=color, linestyle='-', alpha=0.3)
            self._add_event_line(ax, target_row.get(peak_col), color, ':', f'{band} Peak', final_format)
        
        self._add_event_line(ax, target_row.get('First_Observation_Time_MJD'), 'black', '--', 'First Obs', final_format)
            
        self._finalize_plot(ax, target_row, "Multiband", final_format) # FIX APPLIED HERE
        plt.tight_layout()
        plt.show()

    def plot_target_grid(self, target_ids=None, num_cols=3, max_per_figure=12, time_format="MJD"):
        if target_ids is None or target_ids == "all":
            target_ids = self.data_df['Target_ID'].unique().tolist()
            print(f"Plotting grid for ALL {len(target_ids)} targets...")

        # --- NEW: Clear any old, lingering figures from memory ---
        plt.close('all')

        for chunk_start in range(0, len(target_ids), max_per_figure):
            chunk_ids = target_ids[chunk_start : chunk_start + max_per_figure]
            num_targets = len(chunk_ids)
            num_rows = math.ceil(num_targets / num_cols)
            
            # --- NEW: Cap the figure width to 16 inches so it fits on your screen ---
            fig_width = min(16, 5 * num_cols) 
            fig_height = 3.5 * num_rows
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
            axes = np.array(axes).flatten() if num_rows * num_cols > 1 else [axes]

            for i, target_identifier in enumerate(chunk_ids):
                ax = axes[i]
                try:
                    target_row = self._extract_target_row(target_identifier)
                    
                    for band, color, marker in [("ULTRASAT", 'purple', 'o'), ("Visual", 'green', 'X')]:
                        t, m = self._extract_band_data(target_row, band_prefix=band)
                        if len(t) > 0:
                            ax.scatter(t, m, color=color, marker=marker, s=15, label=band)
                            ax.plot(t, m, color=color, linestyle='-', alpha=0.3)
                    
                    ax.set_title(f"Target: {target_identifier}")
                    ax.invert_yaxis()
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set(xlabel="Time (MJD)", ylabel="Magnitude")
                    
                    if i == 0: 
                        ax.legend()
                        
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', transform=ax.transAxes, color='red')
                    ax.set_title(f"Target: {target_identifier} (FAILED)")

            for j in range(num_targets, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            display(fig) # Explicitly tells the notebook to render the image right now
            plt.close(fig) # Safely clears it from memory AFTER it has been rendered


class DatasetStatistics:
    def __init__(self, data_df):
        self.df = data_df

    def _calculate_durations(self):
        time_cols = [col for col in self.df.columns if str(col).startswith("Detection_Time_MJD_")] # list of all columns that start with "Detection_Time_MJD_"
        return self.df[time_cols].max(axis=1) - self.df[time_cols].min(axis=1) # calculates the duration of observations for each target by finding the difference between the maximum and minimum MJD times across all detection time columns.

    def _calculate_time_to_threshold(self, mag_limit=18.0, band="Visual"): # This function calculates how many days it took for each target to reach a specific magnitude limit in either the Visual or ULTRASAT band. It returns a Pandas Series with these durations, which can be used for further analysis or plotting.
        """Calculates days from First Obs to reaching a specific magnitude."""
        results = []
        mag_prefix = "Visual_V_Magnitude_" if band == "Visual" else "ULTRASAT_Magnitude_"
        obs_indices = sorted([int(col.split('_')[-1]) for col in self.df.columns 
                             if col.startswith("Detection_Time_MJD_")])

        for _, row in self.df.iterrows(): # Iterate over each row in the DataFrame
            first_obs = row.get('First_Observation_Time_MJD', np.nan) # Get the time of the first observation for this target
            time_to_reach = np.nan # Initialize the variable that will hold the time it takes to reach the magnitude limit. Start with NaN
            if pd.isna(first_obs): 
                results.append(np.nan); continue

            for idx in obs_indices:
                m = row.get(f"{mag_prefix}{idx}", np.nan)
                t = row.get(f"Detection_Time_MJD_{idx}", np.nan)
                if pd.notna(m) and m <= mag_limit:
                    time_to_reach = t - first_obs
                    break 
            results.append(time_to_reach)
        return pd.Series(results) # Return a Pandas Series containing the time it took for each target to reach the specified magnitude limit in the chosen band. If a target never reached the limit, its value will be NaN.
# if the suppernovae doesnt reach the magnitude limit, it will be assigned a NaN value in the resulting Series. This allows for easy filtering and analysis of only those targets that did reach the threshold.

    def _plot_sensitivity_curve(self, ax, band="Visual", mag_range=(16, 22)): # Plots average time to reach a magnitude threshold vs. the magnitude threshold.
        """Plots Average Time to Reach Mag vs. Mag Threshold."""
        mags = np.linspace(mag_range[0], mag_range[1], 13) # Steps of 0.5 mag
        avg_times = []
        
        for m in mags:
            times = self._calculate_time_to_threshold(mag_limit=m, band=band)
            avg_times.append(times.mean()) # mean() automatically ignores NaNs

        ax.plot(mags, avg_times, marker='o', linestyle='-', color='blue', linewidth=2)
        ax.set_title(f'Avg Time to Reach Mag ({band})', fontweight='bold')
        ax.set_xlabel('Magnitude Limit')
        ax.set_ylabel('Avg Days from First Obs')
        ax.grid(True, linestyle='--', alpha=0.5)

    def plot_summary_dashboard(self, mag_limit=18.0, limit_band="Visual"):
        """Generates the full 4x2 dashboard including the Sensitivity Curve."""
        _, axes = plt.subplots(4, 2, figsize=(12, 20))
        axes = axes.flatten()

        # 1. Sky Map
        axes[0].scatter(self.df['RA (deg)'], self.df['Dec (deg)'], alpha=0.6, s=20, color='royalblue')
        axes[0].set_title('Sky Map: Field of Targets', fontweight='bold')
        axes[0].invert_xaxis()

        # 2-6. Standard Distributions
        self._plot_dist(axes[1], self.df['Redshift'], 'Redshift Distribution', 'z', 'teal') #plots a histogram of the 'Redshift
        self._plot_dist(axes[2], self.df['First_Observation_Time_MJD'], 'First Obs Time', 'MJD', 'darkorange')
        self._plot_dist(axes[3], self._calculate_durations(), 'Total Duration', 'Days', 'crimson')
        self._plot_dist(axes[4], self.df['Peak_Mag_Visual'], 'Visual Peak Mag', 'Mag', 'green', invert_x=True)
        self._plot_dist(axes[5], self.df['Peak_Mag_ULTRASAT'], 'ULTRASAT Peak Mag', 'Mag', 'purple', invert_x=True)

        # 7. Specific Threshold Histogram (e.g., how many reached Mag 18)
        time_to_lim = self._calculate_time_to_threshold(mag_limit, limit_band)
        self._plot_dist(axes[6], time_to_lim, f'Days to Reach <{mag_limit} ({limit_band})', 'Days', 'gold')

        # 8. THE FINAL PLOT: Sensitivity Curve (Continuous 16 to 22)
        self._plot_sensitivity_curve(axes[7], band=limit_band, mag_range=(16, 22))

        plt.tight_layout()
        plt.show()

    def _plot_dist(self, ax, data, title, xlabel, color, invert_x=False):
        clean_data = data.dropna()
        ax.hist(clean_data, bins=20, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.5)
        if invert_x: ax.invert_xaxis()
        if len(clean_data) > 0:
            textstr = f"Mean: {clean_data.mean():.2f}\nVar: {clean_data.var():.2f}"
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', 
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            

def wrap_targets_and_times_for_astroplan(df): # This function takes a DataFrame containing astronomical target data, converts MJD time columns to Astropy Time objects, and creates Astroplan FixedTarget objects for each row. It returns a modified DataFrame with these new Astroplan-ready columns.
    """
    Vectorizes MJD conversions and generates Astroplan FixedTargets.
    """
    print("Upgrading DataFrame to Astroplan-ready objects...")
    df_modified = df.copy()

    # --- A. Vectorized Time Conversion ---
    mjd_cols = [col for col in df_modified.columns if 'MJD' in str(col)]
    for col in mjd_cols:
        valid_mask = df_modified[col].notna()
        if valid_mask.any():
            # 1. Generate the Astropy Time objects
            time_array = Time(df_modified.loc[valid_mask, col].values, format='mjd')
            
            # 2. FORCE Pandas to accept objects: Convert the entire column to a list of Python objects first
            df_modified[col] = df_modified[col].astype('O') 
            
            # 3. Inject the Time objects back into the valid rows
            df_modified.loc[valid_mask, col] = time_array
            
    # --- B. The Clean Function ---
    # We define a tiny helper function right here
    def build_astroplan_target(row):
        coords = SkyCoord(ra=row['RA (deg)'] * u.deg, dec=row['Dec (deg)'] * u.deg) # Create a SkyCoord object using the RA and Dec values from the row, converting them from degrees to the appropriate Astropy units. This object represents the celestial coordinates of the target.
        return FixedTarget(coord=coords, name=row['Target_ID']) # Create and return an Astroplan FixedTarget object using the SkyCoord we just created and the Target_ID from the row as the name. This object can be used directly with Astroplan's scheduling and visibility functions, allowing for seamless integration into observation planning workflows.

    # Apply the function to create a Series of pure FixedTarget objects
    astroplan_objects = df_modified.apply(build_astroplan_target, axis=1)

    # --- C. Clean Column Insertion ---
    if 'Dec (deg)' in df_modified.columns:
        insert_loc = df_modified.columns.get_loc('Dec (deg)') + 1 
        df_modified.insert(insert_loc, 'Astroplan_Target', astroplan_objects) # Insert the new Series of Astroplan FixedTarget objects into the DataFrame as a new column named 'Astroplan_Target'. 
    else:
        df_modified['Astroplan_Target'] = astroplan_objects
        
    print("✅ table successfully built!")
    return df_modified