import os
import pandas as pd
from core.config.settings import setup_libraries, ALL_TELESCOPES
from core.observatories.observatory import ObservationCampaign
from core.observatories.windows_visualization import ScheduleVisualizer
from core.dataprep_and_adjustment.Data_raw_adjustments import load_raw_data, add_peak_and_first_observation_data
from core.CampaignStrategy import (
    combine_target_windows, sort_combined_by_trigger_time,
    simulate_telescope_scheduling_fast, plot_delay_distribution,
    plot_comprehensive_efficiency, plot_scheduling_dashboard, plot_advanced_funnel
)

def main():
    # Setup warnings and libraries
    setup_libraries()

    # 1. Read the path from the text file
    try:
        with open("input_path.txt", "r") as file:
            input_csv_path = file.read().strip() 
    except FileNotFoundError:
        print("Error: Could not find 'input_path.txt'.")
        return

    # Define output paths
    output_csv_path = os.path.join("data", "enriched_lightcurves.csv")
    output_pkl_path = os.path.join("data", "master_windows.pkl")

    # ==========================================
    # Phase 1: Data Enrichment (WITH CACHE CHECK)
    # ==========================================
    if os.path.exists(output_csv_path):
        print(f"⏩ Found existing enriched data. Loading directly from: {output_csv_path}")
        df_enriched = pd.read_csv(output_csv_path)
    else:
        print(f"Loading raw data from: {input_csv_path}")
        df_lightcurves = load_raw_data(input_csv_path)

        print("Extracting peak magnitudes and first observation times...")
        df_enriched = add_peak_and_first_observation_data(df_lightcurves)

        # Save the NEW enriched dataframe
        df_enriched.to_csv(output_csv_path, index=False)
        print(f"✅ Enriched data successfully processed and saved to: {output_csv_path}")

    # ==========================================
    # Phase 2: Telescope Network Check
    # ==========================================
    print("\n--- Telescope Network Initialized ---")
    
    for tel in ALL_TELESCOPES:
        print(f"\n==============================")
        print(f" 🔭 Observatory: {tel.name}")
        print(f"==============================")
        
        lon_val = tel.location.lon.to_string(sep='dms', precision=0)
        lat_val = tel.location.lat.to_string(sep='dms', precision=0)
        elevation = tel.location.height.value
        
        print(f" Location:")
        print(f"   Longitude: {lon_val}")
        print(f"   Latitude:  {lat_val}")
        print(f"   Elevation: {elevation} m")
        print(f"   Timezone:  {tel.tz.zone}")
        
        print(f"\n Capabilities:")
        print(f"   Limiting Mag:   {tel.limiting_mag}")
        print(f"   Seeing Median:  {tel.seeing_median}\" (Limit: {tel.seeing_limit}\")")
        print(f"   Min Obs Window: {tel.min_obs_window} hour(s)")
        
        print(f"\n Constraints:")
        print(f"   Sun Horizon:     {tel.sun_horizon}")
        print(f"   Airmass Horizon: {tel.airmass_horizon}")
        print(f"   Moon Safe Dist:  {tel.moon_base_dist}° (New) to {tel.moon_max_dist}° (Full)")
        
        print(f"\n Weather Profile (Clear Sky Probability):")
        weather_str = " | ".join([f"M{month}:{chance:.2f}" for month, chance in tel.weather_profile.items()])
        print(f"   {weather_str}")
        print(f"==============================\n")

    # ==========================================
    # Phase 3 & 4: Visibility & Weather (WITH CACHE CHECK)
    # ==========================================
    if os.path.exists(output_pkl_path):
        print(f"⏩ Found existing master windows. Loading directly from: {output_pkl_path}")
        df_master_windows = pd.read_pickle(output_pkl_path)
    else:
        print("--- Initializing Observation Campaign ---")
        campaign = ObservationCampaign(ALL_TELESCOPES)
        
        print("🔭 Calculating deep visibility windows (Sun, Moon, Airmass)...")
        df_master_windows = campaign.calculate_observation_windows(df_enriched)

        # Find simulation boundaries dynamically from the time columns
        time_cols = [c for c in df_enriched.columns if str(c).startswith('Detection_Time_MJD_')]
        global_start = df_enriched[time_cols].min().min() - 10
        global_end = df_enriched[time_cols].max().max() + 30

        print("\n☁️ Applying Monte Carlo weather patterns...")
        df_master_windows = campaign.apply_weather_simulation(
            df_master_windows, 
            global_start, 
            global_end
        )

        # Save as Pickle so the Python lists/tuples of windows stay intact
        df_master_windows.to_pickle(output_pkl_path)
        print(f"\n✅ Global campaign complete! Master windows saved to: {output_pkl_path}")

    # ==========================================
    # Phase 5: Visualization
    # ==========================================
    # NOTE: Since this is in main.py, it will pop open a browser tab every time you run the script.
    print("\n📊 Rendering interactive Schedule Visualizer...")
    ScheduleVisualizer.plot_funnel(df_master_windows, df_enriched)

    df_grouped = combine_target_windows(df_master_windows)
    df_sorted = sort_combined_by_trigger_time(df_grouped, df_enriched)

    _, scheduled_lists = simulate_telescope_scheduling_fast(df_sorted)

    plot_delay_distribution(scheduled_lists,  'Windows_Mag_Sun')
    plot_comprehensive_efficiency(scheduled_lists, 'Windows_Final_Weather_Simulated')
    plot_scheduling_dashboard(scheduled_lists)
    plot_advanced_funnel(df_master_windows, df_enriched, scheduled_lists)

if __name__ == "__main__":
    main()