import warnings
import astropy.units as u
from astroplan import download_IERS_A
from core.observatories.observatory import Observatory

# ==========================================
# 1. Environment Setup
# ==========================================
def setup_libraries():
    """Downloads necessary astronomical tables and suppresses annoying warnings."""
    warnings.filterwarnings('ignore', category=Warning)
    # download_IERS_A() # Uncomment this if astropy ever complains about missing time data
    print("Libraries configured and ready.")


# ==========================================
# 2. Weather Profiles
# ==========================================
mast_weather = {1: 0.52, 2: 0.56, 3: 0.55, 4: 0.51, 5: 0.77, 6: 0.89, 7: 0.92, 8: 0.90, 9: 0.76, 10: 0.67, 11: 0.60, 12: 0.64}
soxs_weather = {1: 0.85, 2: 0.88, 3: 0.86, 4: 0.82, 5: 0.78, 6: 0.75, 7: 0.75, 8: 0.79, 9: 0.84, 10: 0.86, 11: 0.88, 12: 0.85}
wilds_weather = {1: 0.67, 2: 0.70, 3: 0.77, 4: 0.84, 5: 0.93, 6: 0.90, 7: 0.45, 8: 0.50, 9: 0.80, 10: 0.87, 11: 0.76, 12: 0.66}


# ==========================================
# 3. Observatory Instantiations
# ==========================================
MAST = Observatory(
    name="MAST", 
    lon_str="35d02m00s", lat_str="+30d03m00s", elevation_m=400, tz_str="Asia/Jerusalem", 
    limiting_mag=25, weather_profile=mast_weather, 
    sun_horizon=-15*u.deg, airmass_horizon=30*u.deg, moon_base_dist=20.0, moon_max_dist=70.0,
    seeing_median=1.3, seeing_limit=2.5,
    min_obs_window=1  
)

SOXS = Observatory(
    name="SOXS", 
    lon_str="-70d44m00s", lat_str="-29d15m00s", elevation_m=2400, tz_str="America/Santiago", 
    limiting_mag=25, weather_profile=soxs_weather, 
    sun_horizon=-15*u.deg, airmass_horizon=30*u.deg, moon_base_dist=20.0, moon_max_dist=70.0,
    seeing_median=0.8, seeing_limit=1.5,
    min_obs_window=1  
)

WILDS = Observatory(
    name="WILDS", 
    lon_str="-111d44m25s", lat_str="+34d44m40s", elevation_m=2360, tz_str="America/Phoenix", 
    limiting_mag=25, weather_profile=wilds_weather,   
    sun_horizon=-15*u.deg, airmass_horizon=30*u.deg, moon_base_dist=20.0, moon_max_dist=70.0,
    seeing_median=1.0, seeing_limit=2.0,
    min_obs_window=1 
)

# A master list for easily iterating over all telescopes in your notebooks
ALL_TELESCOPES = [MAST, SOXS, WILDS]