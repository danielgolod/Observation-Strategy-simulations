import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget



class Target:
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