
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import numpy as np
from astropy.time import Time # <--- NEW IMPORT

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
        else:
            raise TypeError("Target identifier must be a string or a Pandas Series row.")

    def _extract_band_data(self, target_row, band_prefix="ULTRASAT"):
        times = []
        mags = []
        
        detection_indices = [int(col.split('_')[-1]) for col in target_row.index 
                             if str(col).startswith("Detection_Time_MJD_")]
        detection_indices.sort()
        
        for idx in detection_indices:
            time_col = f"Detection_Time_MJD_{idx}"
            if band_prefix == "ULTRASAT":
                mag_col = f"ULTRASAT_Magnitude_{idx}"
            elif band_prefix == "Visual":
                mag_col = f"Visual_V_Magnitude_{idx}"
            else:
                raise ValueError("Supported band prefixes are 'ULTRASAT' or 'Visual'")

            time_val = target_row[time_col]
            mag_val = target_row[mag_col]
            
            if pd.notna(time_val) and pd.notna(mag_val):
                times.append(time_val)
                mags.append(mag_val)
        
        return np.array(times), np.array(mags)

    # --- UPDATED: Now calculates relative positions for the dashed lines ---
    def _add_event_line(self, time_val, color, linestyle, label, time_format, peak_time=None):
        if pd.isna(time_val):
            return
            
        if time_format == "UTC":
            time_val = Time(time_val, format='mjd').datetime
        elif time_format in ["relative_days", "relative_hours"]:
            if pd.isna(peak_time):
                return # Safety catch if peak data is missing
            time_val = time_val - peak_time
            if time_format == "relative_hours":
                time_val = time_val * 24
                
        plt.axvline(x=time_val, color=color, linestyle=linestyle, label=label, alpha=0.7, linewidth=1.5)

    # --- UPDATED: Dynamic labels for the axes ---
    def _finalize_plot(self, target_row, band_name, time_format="MJD"):
        target_id = target_row['Target_ID']
        plt.title(f"Light Curve for {target_id} ({band_name})")
        
        if time_format == "UTC":
            plt.xlabel("Time (UTC)")
            plt.xticks(rotation=45) 
        elif time_format == "relative_days":
            plt.xlabel("Time since Peak (Days)")
        elif time_format == "relative_hours":
            plt.xlabel("Time since Peak (Hours)")
        else:
            plt.xlabel("Time (MJD)")
            
        plt.ylabel("Magnitude (Apparent)")
        plt.gca().invert_yaxis() 
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ultrasat_lc(self, target_identifier, time_format="MJD"):
        target_row = self._extract_target_row(target_identifier)
        times, mags = self._extract_band_data(target_row, band_prefix="ULTRASAT")
        
        peak_time = target_row.get('Peak_Time_MJD_ULTRASAT', np.nan)
        
        # --- TIME CONVERSION LOGIC ---
        if time_format == "UTC":
            times = Time(times, format='mjd').datetime
        elif time_format in ["relative_days", "relative_hours"]:
            if pd.notna(peak_time):
                times = times - peak_time
                if time_format == "relative_hours":
                    times = times * 24
            else:
                print("Warning: No peak time found. Defaulting to MJD.")
                time_format = "MJD"
                
        plt.figure(figsize=(10, 5))
        plt.scatter(times, mags, color='purple', marker='o', s=20, label='ULTRASAT Data')
        plt.plot(times, mags, color='purple', linestyle='-', alpha=0.4)
        
        if 'First_Observation_Time_MJD' in target_row.index:
            self._add_event_line(target_row['First_Observation_Time_MJD'], 'black', '--', 'First Obs', time_format, peak_time)
        if 'Peak_Time_MJD_ULTRASAT' in target_row.index:
            self._add_event_line(target_row['Peak_Time_MJD_ULTRASAT'], 'purple', ':', 'ULTRASAT Peak', time_format, peak_time)
            
        self._finalize_plot(target_row, "ULTRASAT Band", time_format)

    def plot_visual_lc(self, target_identifier, time_format="MJD"):
        target_row = self._extract_target_row(target_identifier)
        times, mags = self._extract_band_data(target_row, band_prefix="Visual")
        
        peak_time = target_row.get('Peak_Time_MJD_Visual', np.nan)
        
        # --- TIME CONVERSION LOGIC ---
        if time_format == "UTC":
            times = Time(times, format='mjd').datetime
        elif time_format in ["relative_days", "relative_hours"]:
            if pd.notna(peak_time):
                times = times - peak_time
                if time_format == "relative_hours":
                    times = times * 24
            else:
                print("Warning: No peak time found. Defaulting to MJD.")
                time_format = "MJD"
                
        plt.figure(figsize=(10, 5))
        plt.scatter(times, mags, color='green', marker='o', s=20, label='Visual V Data')
        plt.plot(times, mags, color='green', linestyle='-', alpha=0.4)
        
        if 'First_Observation_Time_MJD' in target_row.index:
            self._add_event_line(target_row['First_Observation_Time_MJD'], 'black', '--', 'First Obs', time_format, peak_time)
        if 'Peak_Time_MJD_Visual' in target_row.index:
            self._add_event_line(target_row['Peak_Time_MJD_Visual'], 'green', ':', 'Visual Peak', time_format, peak_time)
            
        self._finalize_plot(target_row, "Visual Band", time_format)

    def plot_multiband_lc(self, target_identifier, time_format="MJD"):
        target_row = self._extract_target_row(target_identifier)
        
        # Override relative formats for Multiband
        if time_format in ["relative_days", "relative_hours"]:
            print("Note: Relative time plotting is disabled for multiband curves. Defaulting to MJD.")
            time_format = "MJD"
            
        t_us, m_us = self._extract_band_data(target_row, band_prefix="ULTRASAT")
        t_v, m_v = self._extract_band_data(target_row, band_prefix="Visual")
        
        if time_format == "UTC":
            t_us = Time(t_us, format='mjd').datetime
            t_v = Time(t_v, format='mjd').datetime
            
        plt.figure(figsize=(10, 6))
        plt.scatter(t_us, m_us, color='purple', marker='o', s=25, label='ULTRASAT Data')
        plt.plot(t_us, m_us, color='purple', linestyle='-', alpha=0.3)
        plt.scatter(t_v, m_v, color='green', marker='X', s=25, label='Visual V Data')
        plt.plot(t_v, m_v, color='green', linestyle='-', alpha=0.3)
        
        if 'First_Observation_Time_MJD' in target_row.index:
            self._add_event_line(target_row['First_Observation_Time_MJD'], 'black', '--', 'First Obs', time_format)
        if 'Peak_Time_MJD_ULTRASAT' in target_row.index:
            self._add_event_line(target_row['Peak_Time_MJD_ULTRASAT'], 'purple', ':', 'ULTRASAT Peak', time_format)
        if 'Peak_Time_MJD_Visual' in target_row.index:
            self._add_event_line(target_row['Peak_Time_MJD_Visual'], 'green', ':', 'Visual Peak', time_format)
            
        self._finalize_plot(target_row, "Multiband", time_format)

class StatisticsRandom:
    def __init__(self, csv_path):
        """Initializes the class and loads the Random Targets CSV."""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # Present visibility columns
        self.mast_col = 'MAST_Visible'
        self.soxs_col = 'SOXS_Visible'
        self.wilds_col = 'WILDS_Visible'
        self.visibility_columns = [self.mast_col, self.soxs_col, self.wilds_col]
        
        # Window visibility columns (Update these if you changed the names in your main script!)
        self.mast_win_col = 'MAST_Obs_Windows_48h'
        self.soxs_win_col = 'SOXS_Obs_Windows_48h'
        self.wilds_win_col = 'WILDS_Obs_Windows_48h'

    def print_report(self):
        """Calculates and prints the detection statistics (Snapshots)."""
        print(f"Loading Random Survey data from {self.csv_path}...\n")
        total_targets = len(self.df)
        if total_targets == 0:
            print("No targets found in the CSV!")
            return

        # --- 1. Individual Telescope Calculations ---
        mast_count = self.df[self.mast_col].sum()
        mast_pct = (mast_count / total_targets) * 100
        
        soxs_count = self.df[self.soxs_col].sum()
        soxs_pct = (soxs_count / total_targets) * 100
        
        wilds_count = self.df[self.wilds_col].sum()
        wilds_pct = (wilds_count / total_targets) * 100

        # --- 2. Overall Network Calculations ---
        detected_mask = self.df[self.visibility_columns].any(axis=1)
        detected_count = detected_mask.sum()
        not_detected_count = total_targets - detected_count
        
        detected_pct = (detected_count / total_targets) * 100
        not_detected_pct = (not_detected_count / total_targets) * 100
        
        # --- 3. Print Report ---
        print("=== Random Survey Visibility Statistics (Snapshots) ===")
        print(f"Total Targets Analyzed: {total_targets}\n")
        
        print("--- Individual Telescope Performance ---")
        print(f"MAST (Israel):   {mast_count} detected ({mast_pct:.1f}%)")
        print(f"SOXS (Chile):    {soxs_count} detected ({soxs_pct:.1f}%)")
        print(f"WILDS (USA):     {wilds_count} detected ({wilds_pct:.1f}%)\n")
        
        print("--- Overall Network Performance ---")
        print(f"Total Detected (>= 1 telescope): {detected_count} ({detected_pct:.1f}%)")
        print(f"Total Missed (0 telescopes):     {not_detected_count} ({not_detected_pct:.1f}%)")
        print("=====================================================\n")

    def print_window_report(self):
        """Calculates and prints the detection statistics based on Observation Windows."""
        print(f"Loading Random Survey data for Time Window Analysis...\n")
        total_targets = len(self.df)
        if total_targets == 0:
            print("No targets found in the CSV!")
            return

        # Helper function: When Pandas saves empty lists to a CSV, they become the string '[]'.
        def has_window(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and val.strip() != '[]':
                return True
            if isinstance(val, list) and len(val) > 0:
                return True
            return False

        # --- 1. Create Boolean Masks for the Windows ---
        mast_mask = self.df[self.mast_win_col].apply(has_window)
        soxs_mask = self.df[self.soxs_win_col].apply(has_window)
        wilds_mask = self.df[self.wilds_win_col].apply(has_window)

        # --- 2. Individual Telescope Calculations ---
        mast_count = mast_mask.sum()
        mast_pct = (mast_count / total_targets) * 100
        
        soxs_count = soxs_mask.sum()
        soxs_pct = (soxs_count / total_targets) * 100
        
        wilds_count = wilds_mask.sum()
        wilds_pct = (wilds_count / total_targets) * 100

        # --- 3. Overall Network Calculations ---
        # Combine the masks using | (OR). True if ANY telescope has a window.
        detected_mask = mast_mask | soxs_mask | wilds_mask
        detected_count = detected_mask.sum()
        not_detected_count = total_targets - detected_count
        
        detected_pct = (detected_count / total_targets) * 100
        not_detected_pct = (not_detected_count / total_targets) * 100
        
        # --- 4. Print Report ---
        print("=== Random Survey Visibility Statistics (Time Windows) ===")
        print(f"Total Targets Analyzed: {total_targets}\n")
        
        print(f"--- Individual Telescope Performance (Has >= 1 Window) ---")
        print(f"MAST (Israel):   {mast_count} detected ({mast_pct:.1f}%)")
        print(f"SOXS (Chile):    {soxs_count} detected ({soxs_pct:.1f}%)")
        print(f"WILDS (USA):     {wilds_count} detected ({wilds_pct:.1f}%)\n")
        
        print("--- Overall Network Performance ---")
        print(f"Total Detected (>= 1 telescope): {detected_count} ({detected_pct:.1f}%)")
        print(f"Total Missed (0 telescopes):     {not_detected_count} ({not_detected_pct:.1f}%)")
        print("==========================================================\n")


        # --- How to test it at the bottom of the file ---
if __name__ == "__main__":
    ultrasat_stats = StatisticsULTRASAT("ultrasat_visibility_results.csv")
    ultrasat_stats.print_report()
    ultrasat_stats.print_window_report()


    # Test the Random Class
    random_stats = StatisticsRandom("survey_results_random.csv")
    random_stats.print_report()
    random_stats.print_window_report()