import pandas as pd

class StatisticsULTRASAT:
    def __init__(self, csv_path):
        """Initializes the class and loads the ULTRASAT CSV."""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        self.visibility_columns = [
            'MAST_Visible_Now', 'MAST_Visible_Future',
            'SOXS_Visible_Now', 'SOXS_Visible_Future',
            'WILDS_Visible_Now', 'WILDS_Visible_Future'
        ]

    def print_report(self):
        """Calculates and prints the detection statistics (Snapshots)."""
        print(f"Loading ULTRASAT data from {self.csv_path}...\n")
        total_targets = len(self.df)
        if total_targets == 0:
            print("No targets found in the CSV!")
            return

        # --- 1. Individual Telescope Calculations ---
        # MAST (True if visible now OR future)
        mast_mask = self.df[['MAST_Visible_Now', 'MAST_Visible_Future']].any(axis=1)
        mast_count = mast_mask.sum()
        mast_pct = (mast_count / total_targets) * 100
        
        # SOXS
        soxs_mask = self.df[['SOXS_Visible_Now', 'SOXS_Visible_Future']].any(axis=1)
        soxs_count = soxs_mask.sum()
        soxs_pct = (soxs_count / total_targets) * 100
        
        # WILDS
        wilds_mask = self.df[['WILDS_Visible_Now', 'WILDS_Visible_Future']].any(axis=1)
        wilds_count = wilds_mask.sum()
        wilds_pct = (wilds_count / total_targets) * 100

        # --- 2. Overall Network Calculations ---
        # True if ANY of the 6 columns are True
        detected_mask = self.df[self.visibility_columns].any(axis=1)
        detected_count = detected_mask.sum()
        not_detected_count = total_targets - detected_count
        
        detected_pct = (detected_count / total_targets) * 100
        not_detected_pct = (not_detected_count / total_targets) * 100
        
        # --- 3. Print Report ---
        print("=== ULTRASAT Survey Visibility Statistics (Snapshots) ===")
        print(f"Total Targets Analyzed: {total_targets}\n")
        
        print("--- Individual Telescope Performance (Present or Future) ---")
        print(f"MAST (Israel):   {mast_count} detected ({mast_pct:.1f}%)")
        print(f"SOXS (Chile):    {soxs_count} detected ({soxs_pct:.1f}%)")
        print(f"WILDS (USA):     {wilds_count} detected ({wilds_pct:.1f}%)\n")
        
        print("--- Overall Network Performance ---")
        print(f"Total Detected (>= 1 telescope): {detected_count} ({detected_pct:.1f}%)")
        print(f"Total Missed (0 telescopes):     {not_detected_count} ({not_detected_pct:.1f}%)")
        print("=========================================================\n")

    def print_window_report(self):
        """Calculates and prints the detection statistics based on Observation Windows."""
        print(f"Loading ULTRASAT data for Time Window Analysis...\n")
        total_targets = len(self.df)
        if total_targets == 0:
            print("No targets found in the CSV!")
            return

        # Helper function: When Pandas saves empty lists to a CSV, they become the string '[]'.
        # This returns True only if the target has an actual window with numbers inside.
        def has_window(val):
            if pd.isna(val):
                return False
            if isinstance(val, str) and val.strip() != '[]':
                return True
            if isinstance(val, list) and len(val) > 0:
                return True
            return False

        # --- 1. Create Boolean Masks for the Windows ---
        # UPDATE THESE COLUMN NAMES if you named them differently in your simulation script!
        # (e.g., if you named them 'MAST_Obs_Windows_48h', change it here too)
        mast_mask = self.df['MAST_Obs_Windows_48h'].apply(has_window)
        soxs_mask = self.df['SOXS_Obs_Windows_48h'].apply(has_window)
        wilds_mask = self.df['WILDS_Obs_Windows_48h'].apply(has_window)

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
        print("=== ULTRASAT Survey Visibility Statistics (Time Windows) ===")
        print(f"Total Targets Analyzed: {total_targets}\n")
        
        print(f"--- Individual Telescope Performance (Has >= 1 Window) ---")
        print(f"MAST (Israel):   {mast_count} detected ({mast_pct:.1f}%)")
        print(f"SOXS (Chile):    {soxs_count} detected ({soxs_pct:.1f}%)")
        print(f"WILDS (USA):     {wilds_count} detected ({wilds_pct:.1f}%)\n")
        
        print("--- Overall Network Performance ---")
        print(f"Total Detected (>= 1 telescope): {detected_count} ({detected_pct:.1f}%)")
        print(f"Total Missed (0 telescopes):     {not_detected_count} ({not_detected_pct:.1f}%)")
        print("============================================================\n")


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