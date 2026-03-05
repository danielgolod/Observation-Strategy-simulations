# core/observatories/observatory.py
import astropy.units as u
from astropy.coordinates import EarthLocation
from astroplan import Observer
from pytz import timezone

class Observatory:
    def __init__(self, observer_obj):
        """Initializes with a pre-built astroplan Observer object."""
        self.observer = observer_obj

    @classmethod
    def from_known_site(cls, site_name):
        """Creates an observatory from astroplan's built-in database."""
        return cls(Observer.at_site(site_name))

    @classmethod
    def from_custom_location(cls, name, lon_str, lat_str, elevation_m, tz_str):
        """Creates a custom observatory using string coordinates from the docs."""
        location = EarthLocation.from_geodetic(lon_str, lat_str, elevation_m * u.m)
        observer = Observer(name=name, location=location, timezone=timezone(tz_str))
        return cls(observer)

    def can_see(self, target, time, min_alt=30*u.deg):
        """Checks if the target is visible at a specific global time."""
        is_night = self.observer.is_night(time, horizon=-18*u.deg)
        is_up = self.observer.target_is_up(time, target.astroplan_target, horizon=min_alt)
        return is_night and is_up

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