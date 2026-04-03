import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
from astropy.coordinates import EarthLocation
from pytz import timezone

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from astroplan import Observer
from pytz import timezone

class Observatory:
    # --- CHANGE 1: Changed __init__ to accept the 'observer' object directly ---
    def __init__(self, observer, limiting_mag=None, min_obs_window=1.0, weather_profile=None):
        self.observer = observer  # The Astroplan Observer object is stored here
        self.name = observer.name # Extracting name just in case you need it later
        self.limiting_mag = limiting_mag
        self.min_obs_window = min_obs_window
        # --- CHANGE 2: The class now securely holds its weather data ---
        self.weather_profile = weather_profile 

    # --- CHANGE 3: Added weather_profile to the arguments and return ---
    @classmethod
    def from_known_site(cls, site_name, limiting_mag=19.5, min_obs_window=1.0, weather_profile=None):
        """Creates an observatory from astroplan's built-in database."""
        return cls(Observer.at_site(site_name), limiting_mag, min_obs_window, weather_profile)

    # --- CHANGE 4: Added weather_profile to the arguments and return ---
    @classmethod
    def from_custom_location(cls, name, lon_str, lat_str, elevation_m, tz_str, limiting_mag=19.5, min_obs_window=1.0, weather_profile=None):
        """Creates a custom observatory using string coordinates from the docs."""
        location = EarthLocation.from_geodetic(lon_str, lat_str, elevation_m * u.m)
        observer = Observer(name=name, location=location, timezone=timezone(tz_str))
        
        # We pass the completed observer and the weather_profile to __init__
        return cls(observer, limiting_mag, min_obs_window, weather_profile)

    def can_see(self, target, time, min_alt=30*u.deg):
        """Checks if the target is visible at a specific global time."""
        is_night = self.observer.is_night(time, horizon=-18*u.deg)
        is_up = self.observer.target_is_up(time, target.astroplan_target, horizon=min_alt)
        return is_night and is_up
    
    def can_see_in_future(self, target, time, min_alt=30*u.deg, future_time_window=0):
        """Checks if the target will be visible at any time within the next future_time_window hours."""
        future_time = time + future_time_window * u.hour
        return self.can_see(target, future_time, min_alt=min_alt)
    
    def get_observation_windows(self, target, start_time, window_days=2, step_minutes=15, min_alt=30*u.deg):
        """
        Calculates all continuous observing windows for a target over a given timeframe.
        Returns a list of tuples containing the (start_hour, end_hour) relative to start_time.
        """
        total_minutes = window_days * 24 * 60
        num_steps = int(total_minutes / step_minutes)
        time_offsets = np.arange(num_steps + 1) * step_minutes * u.min
        
        time_grid = start_time + time_offsets

        is_night = self.observer.is_night(time_grid, horizon=-18*u.deg)
        is_up = self.observer.target_is_up(time_grid, target.astroplan_target, horizon=min_alt)
        
        visibility_mask = is_night & is_up 

        padded_mask = np.concatenate(([False], visibility_mask, [False]))
        changes = np.diff(padded_mask.astype(int))
        
        start_indices = np.where(changes == 1)[0]
        end_indices = np.where(changes == -1)[0] - 1 

        windows = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            start_hr = time_offsets[start_idx].to(u.hour).value
            end_hr = time_offsets[end_idx].to(u.hour).value
            
            # --- NO FILTER: Add every single window, no matter how short ---
            windows.append((round(start_hr, 2), round(end_hr, 2)))

        return windows
        


    def get_snr_parameters(self, target, time):
        """Extracts the physical parameters needed for the radiometric SNR equation."""
        
        # 1. Target Position and Airmass
        target_altaz = self.observer.altaz(time, target.astroplan_target)
        airmass = target_altaz.secz.value # Extracts the float value of sec(zenith)
        
        # 2. Moon Parameters (Noise Contributors)
        moon_illum = self.observer.moon_illumination(time)
        moon_pos = self.observer.moon_altaz(time)
        moon_sep = moon_pos.separation(target_altaz).deg # Distance from target
        moon_alt = moon_pos.alt.deg # Is the moon even above the horizon?
        
        # 3. Sun Parameters (Twilight Contributors)
        sun_alt = self.observer.sun_altaz(time).alt.deg
        
        return {
            "airmass": round(airmass, 3),
            "moon_illumination": round(moon_illum, 3), # 0.0 (New) to 1.0 (Full)
            "moon_separation_deg": round(moon_sep, 2),
            "moon_altitude_deg": round(moon_alt, 2),
            "sun_altitude_deg": round(sun_alt, 2)
        }