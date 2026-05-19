
from datetime import datetime
from pytz import timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm

import astropy.units as u
from astropy.time import Time
import astropy.coordinates as coord
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astroplan import Observer, FixedTarget

# ==========================================
# 2. OBSERVATORY CLASS
# ==========================================
class Observatory:

    def __init__(self, name, lon_str, lat_str, elevation_m, tz_str, limiting_mag, 
                 weather_profile, min_obs_window=1.0, 
                 sun_horizon=-15*u.deg, airmass_horizon=30*u.deg, 
                 moon_base_dist=20.0, moon_max_dist=70.0, 
                 seeing_median=1.0, seeing_limit=2.0):
        
        # 1. General Info
        self.name = name
        self.limiting_mag = limiting_mag # mag limit for the observatory 
        self.weather_profile = weather_profile # dict of month (1-12) to clear sky probability (0-1)
        self.min_obs_window = min_obs_window # minimum time for taking spectrum in hours (used in window filtering later)
        
        # 2. Location & Astroplan Setup
        self.tz = timezone(tz_str) # pytz timezone object for local time conversions
        self.location = EarthLocation.from_geodetic(lon_str, lat_str, elevation_m * u.m) # astropy EarthLocation object
        self.observer = Observer(name=name, location=self.location, timezone=timezone(tz_str)) # astroplan Observer object for visibility calculations
        
        self.sun_horizon = sun_horizon # the angle to the sun below the horizon to consider it "dark" 
        self.airmass_horizon = airmass_horizon # the minimum altitude (in degrees) above the horizon for a target to be considered observable
        self.moon_base_dist = moon_base_dist # the minimum safe distance from the moon in degrees when the moon is new
        self.moon_max_dist = moon_max_dist # the maximum safe distance from the moon in degrees when the moon is full
        self.seeing_median = seeing_median # the median seeing value in arcseconds for this observatory 
        self.seeing_limit = seeing_limit # the maximum acceptable seeing in arcseconds for an observation to be considered successful
    
    def get_weather_chance(self, month):
        # Directly grab the exact month from your assigned weather profile
        return self.weather_profile[month]

    
    def generate_weather_windows(self, start_mjd, end_mjd):
        
        
        # 1. Create the dense 15-minute grid
        step_days = 15.0 / (24.0 * 60.0)
        time_grid = np.arange(start_mjd, end_mjd + step_days, step_days)
        
        # 2. Ask Astropy which ticks are perfectly dark
        time_objects = Time(time_grid, format='mjd')
        is_dark = self.observer.is_night(time_objects, horizon=self.sun_horizon)
        
        # 3. Load it into Pandas and immediately drop the daylight!
        # Now it goes from a "dense" grid to a "sparse" grid.
        df_grid = pd.DataFrame({'Time_MJD': time_grid, 'Is_Dark': is_dark}) # Create a DataFrame with the time grid and corresponding darkness information
        df_night = df_grid[df_grid['Is_Dark']].copy() # Filter the DataFrame to include only rows where it is dark, creating a new DataFrame for nighttime observations
        
        if df_night.empty:
            return []
            
        # 4. The Pandas Magic (Exactly like the Campaign class!)
        gap_threshold = 20 / (24 * 60) # A gap larger than 20 minutes means a new night
        df_night['time_diff'] = df_night['Time_MJD'].diff()
        df_night['is_new_window'] = (df_night['time_diff'] > gap_threshold) | df_night['time_diff'].isna()
        df_night['window_id'] = df_night['is_new_window'].cumsum()
        
        # 5. Extract the windows and apply the weather dice roll
        clear_weather_windows = []
        self.full_astropy_calendar = [] # <--- NEW: Save ALL nights for the visualizer
        
        for _, group in df_night.groupby('window_id'):
            n_start = group['Time_MJD'].min()
            n_end = group['Time_MJD'].max()
            
            # Save every single night to the new list
            self.full_astropy_calendar.append((n_start, n_end)) 
            
            # The Dice Roll
            month = Time(n_start, format='mjd').datetime.month
            clear_chance = self.get_weather_chance(month)
            
            if np.random.random() < clear_chance:
                clear_weather_windows.append((n_start, n_end))
                
        return clear_weather_windows

    def render_global_map(self, other_telescopes, local_time_str):
        """
        Renders the day/night terminator. 
        Input time is assumed to be the LOCAL time of THIS specific observatory.
        """
        # --- A. Timezone Math ---
        local_dt = datetime.strptime(local_time_str, "%Y-%m-%d %H:%M")
        local_aware = self.tz.localize(local_dt)
        utc_dt = local_aware.astimezone(timezone('UTC'))
        obs_time = Time(utc_dt)
        current_month = utc_dt.month

        # --- B. Build Network Data ---
        all_obs = [self] + other_telescopes
        data = []
        for obs in all_obs:
            obs_local_dt = utc_dt.astimezone(obs.tz)
            
            clear_chance = obs.weather_profile.get(current_month, 0.0)
            
            data.append({
                'Observatory': obs.name,
                'Lat': obs.location.lat.deg,
                'Lon': obs.location.lon.deg,
                'Local Time': obs_local_dt.strftime("%b %d, %H:%M"),
                'Clear Skies': f"{int(clear_chance * 100)}%"
            })
        df_map = pd.DataFrame(data)

        # --- C. Astropy Terminator Math ---
        sun = get_sun(obs_time)
        gast = obs_time.sidereal_time('apparent', 'greenwich')
        subsolar_lon = (sun.ra - gast).wrap_at(180 * u.deg)
        subsolar_lat = sun.dec
        lons = np.linspace(-180, 180, 360)
        
        term_lats = np.degrees(np.arctan(
            -np.cos(np.radians(lons - subsolar_lon.deg)) / np.tan(subsolar_lat.radian + 1e-6)
        ))
        
        if subsolar_lat.deg > 0:
            poly_lons = np.concatenate([lons, [180, -180]])
            poly_lats = np.concatenate([term_lats, [-90, -90]])
        else:
            poly_lons = np.concatenate([lons, [180, -180]])
            poly_lats = np.concatenate([term_lats, [90, 90]])

        # --- D. Build the Plotly Map ---
        fig = px.scatter_geo(
            df_map, lat='Lat', lon='Lon', hover_name='Observatory',
            hover_data={'Local Time': True, 'Clear Skies': True, 'Lat': False, 'Lon': False},
            text='Observatory', color='Observatory', projection="natural earth"
        )

        fig.add_trace(go.Scattergeo(
            lon=poly_lons, lat=poly_lats, fill='toself', fillcolor='rgba(0, 0, 15, 0.4)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False, name='Night'
        ))

        fig.update_traces(
            marker=dict(size=14, line=dict(width=2, color='black')), 
            textposition="bottom center", textfont=dict(size=14, family="Arial Black", color="black"),
            selector=dict(mode='markers+text')
        )

        fig.update_layout(
            title_text=f'Network Coverage: Base Time [{self.name} local: {local_time_str}]<br><sup>UTC Equivalent: {obs_time.iso[:16]}</sup>',
            title_x=0.5, title_font=dict(size=18, color="darkblue", family="Arial Black"),
            showlegend=False,
            geo=dict(
                showland=True, landcolor="rgb(240, 240, 240)", showocean=True, oceancolor="rgb(204, 230, 255)",     
                showcountries=True, countrycolor="rgb(200, 200, 200)", coastlinecolor="rgb(150, 150, 150)", 
                lataxis=dict(showgrid=True, gridcolor="rgb(220, 220, 220)"), lonaxis=dict(showgrid=True, gridcolor="rgb(220, 220, 220)")
            ),
            margin=dict(l=0, r=0, t=75, b=0) 
        )
        fig.show()

# ==========================================
# 3. OBSERVATION CAMPAIGN CLASS
# ==========================================
class ObservationCampaign: 

    @staticmethod
    def extract_windows(df_obs, mask):  
        df_valid = df_obs[mask].copy()
        
        if df_valid.empty:
            return []

        gap_threshold = 20 / (24 * 60)  
        padding_mjd = 7.5 / (24 * 60)

        df_valid['time_diff'] = df_valid['Time_MJD'].diff() # calculates the diff between to adjacent rows in the 'Time_MJD' column and stores it in a new column called 'time_diff' (for the first row, this will be NaN since there's no previous row to compare to) 
        df_valid['is_new_window'] = (df_valid['time_diff'] > gap_threshold) | df_valid['time_diff'].isna() # if the tme diff between two points larger then 20 minutes, it means a new window is starting. This creates a boolean column where True indicates the start of a new window 
        df_valid['window_id'] = df_valid['is_new_window'].cumsum() # just when it sees a True in the 'is_new_window' column, it increments the 'window_id' by 1. This way, all consecutive rows that belong to the same window will have the same 'window_id'.
        
        windows = []
        
        for _, group in df_valid.groupby('window_id'): # we group by the value of the window_id, so each group corresponds to a continuous observing window. We then iterate over these groups to extract the start and end times of each window.
            start_mjd = group['Time_MJD'].min()-padding_mjd # we subtract a small padding from the start time to account for the fact that our time grid is in 15-minute increments. 
            end_mjd = group['Time_MJD'].max()+padding_mjd
            avg_mag = group['Mag_Visual'].mean()

            windows.append((start_mjd, end_mjd, avg_mag))
                
        return windows

    def __init__(self, observatories):
        self.observatories = observatories # imports the list of Observatory objects that are part of this campaign. Each observatory will have its own location, weather profile, and observing constraints, which will be used to calculate the observation windows for each target in the enriched dataframe.

    def calculate_observation_windows(self, df_enriched, step_minutes=15): 
        
        t_cols = [c for c in df_enriched.columns if c.startswith('Detection_Time_MJD_')] # all columns that start with 'Detection_Time_MJD_' (the headers)
        u_cols = [c for c in df_enriched.columns if c.startswith('ULTRASAT_Magnitude_')] # all columns that start with 'ULTRASAT_Magnitude_'
        v_cols = [c for c in df_enriched.columns if c.startswith('Visual_V_Magnitude_')] # all columns that start with 'Visual_V_Magnitude_'

        # This will now hold our lightweight dictionaries instead of giant dataframes
        all_timelines = []

        for index, row in tqdm(df_enriched.iterrows(), total=len(df_enriched), desc="Processing Targets"): # iterate over each target in the dataframe with a progress bar
            target_id = row['Target_ID']
            
            # Generating Astroplan targets just-in-time to keep the dataframe clean
            coords = SkyCoord(ra=row['RA (deg)'] * u.deg, dec=row['Dec (deg)'] * u.deg) # Create a SkyCoord object using the RA and Dec values from the row.
            astro_target = FixedTarget(coord=coords, name=target_id) # Create and return an Astroplan FixedTarget object.
            
            valid_mask = row[t_cols].notna().values # boolean mask to identify which time columns have valid (non-NaN) data for this target. 
            raw_time_data = row[t_cols].values[valid_mask] # creates an array of the valid time data for this target 
            
            if len(raw_time_data) < 2:
                continue 
                
            raw_times = raw_time_data.astype(float) # convert the valid time data to a numpy array of floats (MJD times) for easier numerical processing later on.
            raw_u_mags = row[u_cols].values[valid_mask].astype(float) 
            raw_v_mags = row[v_cols].values[valid_mask].astype(float)
            
            start_mjd = np.min(raw_times) 
            end_mjd = np.max(raw_times) 
            start_time = Time(start_mjd, format='mjd') # Convert the minimum MJD time to an Astropy Time object
            end_time = Time(end_mjd, format='mjd') # Convert the maximum MJD time to an Astropy Time object
            
            total_minutes = (end_time - start_time).to(u.minute).value  # Calculate the total time span in minutes
            num_steps = int(total_minutes / step_minutes)  # Determine how many time steps we need to create
            
            time_grid = start_time + np.arange(num_steps + 1) * step_minutes * u.min # Create a time grid with the specified step size
            grid_mjd = time_grid.mjd 
            
            df_grid = pd.DataFrame({'Time_MJD': grid_mjd})
            df_raw = pd.DataFrame({
                'Time_MJD': raw_times,
                'Mag_ULTRASAT': raw_u_mags,
                'Mag_Visual': raw_v_mags
            })
            
            df_raw = df_raw.sort_values('Time_MJD') # Ensure the raw data is sorted by time for the merge_asof to work correctly
            df_sim = pd.merge_asof(df_grid, df_raw, on='Time_MJD', direction='nearest') # Use merge_asof to align the raw magnitude data with the time grid
            df_sim.insert(0, 'Target_ID', target_id)
            
            # Temporary list for THIS target's dictionaries
            target_obs_list = [] 
            
            for obs in self.observatories:
                df_obs = df_sim.copy()
                df_obs['Observatory'] = obs.name  # Add the current observatory name 
                
                # 1. MAGNITUDE FILTER
                mask_mag = df_obs['Mag_Visual'].values <= obs.limiting_mag 
                times_mag = time_grid[mask_mag] 
                
                # Setup our decoupled masks
                mask_telemetry = np.zeros(len(time_grid), dtype=bool)
                mask_strict_sun = np.zeros(len(time_grid), dtype=bool)
                mask_airmass = np.zeros(len(time_grid), dtype=bool)
                mask_final = np.zeros(len(time_grid), dtype=bool)
                
                telemetry_dict = {}

                # 2. BROAD SUN FILTER (For Plotting: Sunset to Sunrise)
                if len(times_mag) > 0:
                    # horizon=0*u.deg keeps the telemetry running during twilight!
                    telemetry_subset = obs.observer.is_night(times_mag, horizon=0*u.deg)
                    mask_telemetry[mask_mag] = telemetry_subset
                    
                times_telemetry = time_grid[mask_telemetry]

                # 3. CALCULATE FULL-NIGHT TELEMETRY
                if len(times_telemetry) > 0:
                    target_altaz = obs.observer.altaz(times_telemetry, astro_target)
                    moon_altaz = obs.observer.moon_altaz(times_telemetry)
                    sun_altaz = obs.observer.sun_altaz(times_telemetry) # <-- NEW: Capture Sun!
                    
                    telemetry_dict = {
                        'Time_MJD': times_telemetry.mjd.tolist(),
                        'Target_Elevation': target_altaz.alt.deg.tolist(),
                        'Moon_Separation': moon_altaz.separation(target_altaz).deg.tolist(),
                        'Moon_Elevation': moon_altaz.alt.deg.tolist(),
                        'Sun_Elevation': sun_altaz.alt.deg.tolist() # <-- NEW: Save to dictionary!
                    }

                    # 4. STRICT FILTERS FOR ACTUAL OBSERVATION WINDOWS
                    # A. Strict Astronomical Twilight
                    strict_sun_subset = obs.observer.is_night(times_telemetry, horizon=obs.sun_horizon)
                    mask_strict_sun[mask_telemetry] = strict_sun_subset
                    
                    # B. Airmass Limit
                    airmass_subset = obs.observer.target_is_up(times_telemetry, astro_target, horizon=obs.airmass_horizon)
                    mask_airmass[mask_telemetry] = airmass_subset
                    
                    # Combine strict filters to prevent shape mismatches
                    mask_combined = mask_strict_sun & mask_airmass
                    times_combined = time_grid[mask_combined]
                    
                    # C. Moon Separation & Illumination
                    if len(times_combined) > 0:
                        astropy_times_combined = Time(times_combined, format='mjd')
                        
                        target_altaz_subset = obs.observer.altaz(astropy_times_combined, astro_target)
                        moon_altaz_subset = obs.observer.moon_altaz(astropy_times_combined)
                        
                        moon_sep = moon_altaz_subset.separation(target_altaz_subset)
                        moon_illum = obs.observer.moon_illumination(astropy_times_combined)
                        moon_is_down = moon_altaz_subset.alt < 0 * u.deg
                        
                        distance_scale = obs.moon_max_dist - obs.moon_base_dist
                        dynamic_safe_dist = (obs.moon_base_dist + (distance_scale * moon_illum)) * u.deg
                        moon_is_safe_dist = moon_sep > dynamic_safe_dist
                        
                        # Only times that pass Sun, Airmass, AND Moon make it into the final windows
                        mask_final[mask_combined] = moon_is_down | moon_is_safe_dist

                target_summary = {
                    'Target_ID': target_id,
                    'Observatory': obs.name,
                    'Telemetry': telemetry_dict 
                }
                
                # --- FUNNEL LEVELS 1-3: Restore intermediate steps for the Plotly Dashboard ---
                target_summary['Windows_Mag_Only'] = self.extract_windows(df_obs, mask_mag)
                target_summary['Windows_Mag_Sun'] = self.extract_windows(df_obs, mask_strict_sun)
                target_summary['Windows_Mag_Sun_Airmass'] = self.extract_windows(df_obs, mask_combined)

                # --- FUNNEL LEVEL 4: Moon Filtered ---
                target_summary['mask_mag_sun_airmass_moon'] = self.extract_windows(df_obs, mask_final)

                # --- FUNNEL LEVEL 5: Duration Filtered ---
                min_req_mins = obs.min_obs_window * 60 
                
                duration_filtered_windows = [
                    (start, end, mag) for (start, end, mag) in target_summary['mask_mag_sun_airmass_moon'] 
                    if (end - start) * 24 * 60 >= min_req_mins
                ]
                target_summary['mask_mag_sun_airmass_moon_duration'] = duration_filtered_windows
                
                # --- FUNNEL LEVEL 6: Weather Filter (Simulated) ---
                # Placeholder: apply_weather_simulation() fills this column using the
                # observatory-wide clear-night calendar, which is the correct approach.
                # A per-window dice roll here would be wrong because windows on the same
                # night would roll independently, and the result gets overwritten anyway.
                target_summary['Windows_Final_Weather_Simulated'] = duration_filtered_windows

                target_obs_list.append(target_summary)

            # Once the inner observatory loop finishes, add these summaries to the master list
            all_timelines.extend(target_obs_list)

        print("💾 Compiling Master Windows Database...")
        df_master_windows = pd.DataFrame(all_timelines)  

        return df_master_windows
    
    def apply_weather_simulation(self, df_master, global_start_mjd, global_end_mjd):
        print("☁️ Filtering targets against explicit weather blocks...")
        
        new_column = 'Windows_Final_Weather_Simulated'
        df_master[new_column] = df_master['mask_mag_sun_airmass_moon_duration']
        epsilon = 1e-5 

        # Create a dictionary to hold the real calendars for Plotly
        self.saved_weather_calendars = {} 

        for obs in self.observatories:
            # 1. Get the explicit clear night windows for this telescope
            clear_night_windows = obs.generate_weather_windows(global_start_mjd, global_end_mjd)
            
            # Save the calendar to the dictionary!
            self.saved_weather_calendars[obs.name] = clear_night_windows 
            
            obs_mask = df_master['Observatory'] == obs.name
            
            for idx, row in df_master[obs_mask].iterrows():
                target_windows = row['mask_mag_sun_airmass_moon_duration']
                surviving_target_windows = []
                
                for t_win in target_windows:
                    t_start, t_end, t_mag = t_win

                    # A target window survives only if it actually overlaps with a clear night.
                    # Overlap condition: night starts before window ends AND night ends after window starts.
                    is_covered = any(
                        n_start <= t_end and n_end >= t_start
                        for n_start, n_end in clear_night_windows
                    )

                    if is_covered:
                        surviving_target_windows.append(t_win)
                        
                # Overwrite the dataframe cell with the surviving windows
                df_master.at[idx, new_column] = surviving_target_windows
                
        # ---> THE MISSING RETURN IS HERE <---
        # It aligns with the 'for obs in self.observatories:' loop
        return df_master
    

    