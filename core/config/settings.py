# core/config/settings.py
import warnings
from astroplan import download_IERS_A

def setup_libraries():
    """Downloads necessary astronomical tables and suppresses annoying warnings."""
    warnings.filterwarnings('ignore', category=Warning)
    # download_IERS_A() # Uncomment this if astropy ever complains about missing time data
    print("Libraries configured and ready.")