import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget
import pandas as pd
from astropy.time import Time



class Target_random_for_testing:
    def __init__(self, fixed_target_obj):
        """Initializes with a pre-built astroplan FixedTarget object."""
        self.astroplan_target = fixed_target_obj

    @classmethod
    def from_coordinates(cls, name, ra_str, dec_str):
        """Creates a target using string coordinates (e.g., '19h50m47.6s')."""
        coord = SkyCoord(ra_str, dec_str, frame='icrs')
        return cls(FixedTarget(name=name, coord=coord))

    @classmethod
    def from_name(cls, name):
        """Retrieves coordinates automatically using the CDS name resolver."""
        return cls(FixedTarget.from_name(name))
    
    @classmethod
    def generate_random(cls, name="Random_Transient"):
        """Generates evenly distributed coordinates on the sphere."""
        ra = np.random.uniform(0, 360)
        dec = np.degrees(np.arcsin(np.random.uniform(-1, 1)))
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        return cls(FixedTarget(coord=coord, name=name))
    

class Target_ULTRASAT_Wrapped:
    def __init__(self, fixed_target_obj):
        """A clean wrapper to hold real ULTRASAT targets for the observatories."""
        self.astroplan_target = fixed_target_obj

class Targets_ULTRASAT:
    def __init__(self, csv_path):
        """Loads the CSV and automatically adds the real-time column."""
        print(f"Loading ULTRASAT targets from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.add_real_time_column()

    def add_real_time_column(self):
        """Converts MJD to human-readable UTC time."""
        mjd_times = Time(self.data['Peak_Time_MJD'].values, format='mjd')
        self.data['Peak_Real_Time (UTC)'] = mjd_times.iso

    def get_dataframe(self):
        """Returns the fully prepared Pandas DataFrame."""
        return self.data