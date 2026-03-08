import pandas as pd
from astropy.time import Time
from astropy import units as u
import numpy as np

# Import from your new organized folders!
from core.config.settings import setup_libraries
from core.targets.simulation_targets import Target_random_for_testing
from core.observatories.observatory import Observatory

# 1. Run global setups

setup_libraries()

# 2. Initialize Observatories

# MAST/DeepSpec
mast = Observatory.from_custom_location(
    name="MAST", lon_str="35d02m00s", lat_str="+30d03m00s", elevation_m=400, tz_str="Asia/Jerusalem"
)

# NTT/SOXS
soxs = Observatory.from_custom_location(
    name="SOXS", lon_str="-70d44m00s", lat_str="-29d15m00s", elevation_m=2400, tz_str="America/Santiago"
)

# LDT/WILDS
wilds = Observatory.from_custom_location(
    name="WILDS", lon_str="-111d44m25s", lat_str="+34d44m40s", elevation_m=2360, tz_str="America/Phoenix"
)

#Global Parameters
future_time_window = 0 # in hours, for the future visibility check
observation_window_days = 2 # in days, for the continuous window calculation
# --- 3. Generate 1000 Targets & Random Times this wa for testing---

# Set start date to today
start_time = Time('2026-03-04 11:40:00')

# Create an array of 1000 random targets
print("Generating 1000 targets and times...")
targets = [Target_random_for_testing.generate_random(name=f"T_{i}") for i in range(1000)]

# Add a random number of days (between 0 and 365) to the start time for each target
random_days = np.random.uniform(0, 365, 1000)
times = start_time + random_days * u.day

# --- 3. Run the Simulation ---

results = []

print("Running visibility, SNR parameter checks, and 48-hour windows (this may take a minute)...")

for i in range(1000):
    targ = targets[i]
    t = times[i]
    
    # 1. Check basic visibility AND Observation Windows
    mast_vis = mast.can_see(targ, t)
    mast_vis_future = mast.can_see_in_future(targ, t, future_time_window=future_time_window)
    mast_windows = mast.get_observation_windows(targ, t, window_days=observation_window_days)
    
    soxs_vis = soxs.can_see(targ, t)
    soxs_vis_future = soxs.can_see_in_future(targ, t, future_time_window=future_time_window)
    soxs_windows = soxs.get_observation_windows(targ, t, window_days=observation_window_days)
    
    wilds_vis = wilds.can_see(targ, t)
    wilds_vis_future = wilds.can_see_in_future(targ, t, future_time_window=future_time_window)
    wilds_windows = wilds.get_observation_windows(targ, t, window_days=observation_window_days)
    
    # 2. Set up the base row data
    row_data = {
        "Target_ID": targ.astroplan_target.name,
        "RA (deg)": round(targ.astroplan_target.coord.ra.deg, 2),
        "Dec (deg)": round(targ.astroplan_target.coord.dec.deg, 2),
        "Time (UTC)": t.iso[:19],
        "MAST_Visible": mast_vis,
        "MAST_Visible_in_Future": mast_vis_future,
        "MAST_Obs_Windows_48h": mast_windows,
        "SOXS_Visible": soxs_vis,
        "SOXS_Visible_in_Future": soxs_vis_future,
        "SOXS_Obs_Windows_48h": soxs_windows,
        "WILDS_Visible": wilds_vis,
        "WILDS_Visible_in_Future": wilds_vis_future,
        "WILDS_Obs_Windows_48h": wilds_windows
    }
    
    # 3. Add the exact SNR parameters ONLY if the telescope can see the target right now
    if mast_vis:
        params = mast.get_snr_parameters(targ, t)
        for key, value in params.items(): 
            row_data[f"MAST_{key}"] = value
            
    if soxs_vis:
        params = soxs.get_snr_parameters(targ, t)
        for key, value in params.items(): 
            row_data[f"SOXS_{key}"] = value
            
    if wilds_vis:
        params = wilds.get_snr_parameters(targ, t)
        for key, value in params.items(): 
            row_data[f"WILDS_{key}"] = value

    results.append(row_data)

# --- 4. Display the Results ---

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(results)

# Save the results directly to the main folder so statistics.py can find it easily!
output_filename = 'survey_results_random.csv'
df.to_csv(output_filename, index=False)

print("\n--- Simulation Complete ---\n")

# Display a subset of columns for the first few successful MAST detections
mast_successes = df[df['MAST_Visible'] == True]
if not mast_successes.empty:
    print("Example of successful MAST detections with SNR parameters:")
    print(mast_successes[['Target_ID', 'Time (UTC)', 'MAST_airmass', 'MAST_moon_illumination']].head())
else:
    print("No MAST targets were visible in this run.")

print("\n--- Total Immediate Detections over 1 Year ---")
print(f"MAST (Israel):  {df['MAST_Visible'].sum()} targets")
print(f"SOXS (Chile):   {df['SOXS_Visible'].sum()} targets")
print(f"WILDS (USA):    {df['WILDS_Visible'].sum()} targets")

print(f"\n✅ Data successfully saved to: {output_filename}")