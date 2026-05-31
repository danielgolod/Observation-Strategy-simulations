# Observation Strategy Simulations

A simulation framework for optimizing spectroscopic follow-up of ULTRASAT supernova alerts across a multi-telescope network. Given simulated transient light curves, the pipeline calculates when each target is observable at each telescope, applies realistic weather and constraint filtering, and schedules 1-hour spectroscopic slots — producing interactive dashboards that quantify network efficiency and response time.

---


---

## Telescope Network


Weather profiles (monthly clear-sky probability), sun/moon/airmass constraints, and seeing limits are configured independently for each telescope in `core/config/settings.py`.

---

## Project Structure

```
.
├── mainscript.py                        # End-to-end pipeline runner
├── testing.ipynb                        # Interactive exploration notebook
├── input_path.txt                       # Path to the raw input CSV (you set this)
├── requirements.txt
│
├── core/
│   ├── config/
│   │   └── settings.py                  # Observatory definitions, weather profiles, global constants
│   │
│   ├── observatories/
│   │   ├── observatory.py               # Observatory class, ObservationCampaign class
│   │   ├── windows_visualization.py     # ScheduleVisualizer (funnel summary plots)
│   │   └── observatory_night_starlart_sim.py  # Staralt web scraper + clone visualizer
│   │
│   ├── dataprep_and_adjustment/
│   │   └── Data_raw_adjustments.py      # Raw CSV loader, peak/first-obs enrichment
│   │
│   └── CampaignStrategy.py             # Scheduling algorithm + all Plotly dashboards
│
└── data/                                # Auto-created on first run
    ├── enriched_lightcurves.csv         # Cache: enriched target list
    ├── master_windows.pkl               # Cache: visibility + weather windows
    └── staralt_gifs/                    # Staralt validation GIFs
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. All dependencies are standard astronomical Python:

```
numpy, pandas, matplotlib, plotly, ipympl, tqdm, pytz, astropy, astroplan
```

If astropy warns about missing IERS data on first run, uncomment the download line in `core/config/settings.py`:
```python
# from astroplan import download_IERS_A; download_IERS_A()
```

---

## How to Run



The CSV must contain ULTRASAT and Visual V-band light curves in wide format, with columns:
- `Target_ID`, `RA (deg)`, `Dec (deg)`
- `Detection_Time_MJD_0`, `Detection_Time_MJD_1`, ... (time grid per epoch)
- `ULTRASAT_Magnitude_0`, `ULTRASAT_Magnitude_1`, ...
- `Visual_V_Magnitude_0`, `Visual_V_Magnitude_1`, ...

###  Run the full pipeline

```bash
python mainscript.py
```

The script runs in five phases and caches the expensive steps:

| Phase | What it does | Cache file |
|-------|-------------|------------|
| 1 | Load CSV, compute peak magnitude and first-observation time | `data/enriched_lightcurves.csv` |
| 2 | Print telescope network summary | — |
| 3–4 | Calculate visibility windows + apply weather simulation | `data/master_windows.pkl` |
| 5 | Launch interactive Plotly dashboards | — |

On re-runs, phases 1 and 3–4 are skipped if their cache files exist. Delete the cache files to force a full recompute.

### 3. Interactive exploration (notebook)

Open `testing.ipynb` in Jupyter and run cells top to bottom. The notebook mirrors the main pipeline but exposes intermediate outputs for inspection, including the per-target constraint funnel explorer and Staralt validation plots.

---

## Module Reference

### `core/config/settings.py`

Central configuration. Edit this file to change telescopes or constraints.

| Symbol | Description |
|--------|-------------|
| `MAST`, `SOXS`, `WILDS` | Pre-built `Observatory` instances |
| `ALL_TELESCOPES` | List of all three, used throughout the pipeline |
| `setup_libraries()` | Seeds NumPy RNG (seed=42), suppresses warnings |
| `mast_weather`, `soxs_weather`, `wilds_weather` | Monthly clear-sky probability dicts |

---

### `core/observatories/observatory.py`

**`Observatory`** — represents a single telescope site.

| Attribute | Description |
|-----------|-------------|
| `name` | Telescope identifier string |
| `location` | `astropy.EarthLocation` |
| `observer` | `astroplan.Observer` for rise/set/altitude queries |
| `limiting_mag` | Faintest observable Visual magnitude |
| `weather_profile` | Dict mapping month (1–12) to clear probability |
| `sun_horizon` | Sun altitude defining astronomical night (e.g. −15°) |
| `airmass_horizon` | Minimum target altitude above horizon (e.g. 30°) |
| `moon_base_dist` / `moon_max_dist` | Moon exclusion zone: scales linearly with illumination |
| `seeing_median` / `seeing_limit` | Seeing statistics in arcseconds |
| `min_obs_window` | Minimum viable window duration in hours |

Key methods:
- `get_weather_chance(month)` — returns clear probability for a given month
- `generate_weather_windows(start_mjd, end_mjd)` — Monte Carlo clear-night calendar over a date range
- `render_global_map(other_telescopes, local_time_str)` — interactive Plotly world map with day/night terminator

**`ObservationCampaign`** — runs the visibility calculation across all targets and telescopes.

| Method | Description |
|--------|-------------|
| `calculate_observation_windows(df_enriched)` | Builds the 6-level constraint funnel for every target × telescope combination. Returns `df_master_windows`. |
| `apply_weather_simulation(df_master, start_mjd, end_mjd)` | Generates per-observatory clear-night calendars and filters windows against them. Overwrites `Windows_Final_Weather_Simulated`. |

The `df_master_windows` dataframe has one row per (target, observatory) pair, with list columns for each funnel level:

```
Windows_Mag_Only → Windows_Mag_Sun → Windows_Mag_Sun_Airmass →
mask_mag_sun_airmass_moon → mask_mag_sun_airmass_moon_duration →
Windows_Final_Weather_Simulated
```

Each list entry is a tuple `(start_mjd, end_mjd, avg_visual_mag)`.

---

### `core/dataprep_and_adjustment/Data_raw_adjustments.py`

| Function | Description |
|----------|-------------|
| `load_raw_data(csv_path)` | Reads the wide-format CSV, validates columns, returns `df_lightcurves` |
| `add_peak_and_first_observation_data(df)` | Adds `Peak_Mag_ULTRASAT`, `Peak_Mag_Visual`, `First_Observation_Time_MJD`, and `Peak_Time_MJD` columns. Returns `df_enriched`. |

---

### `core/CampaignStrategy.py`

Scheduling logic and all Plotly output functions.

| Function | Description |
|----------|-------------|
| `combine_target_windows(df_master)` | Merges per-observatory windows for each target into a single sorted list. Appends observatory name as 4th element of each tuple. |
| `sort_combined_by_trigger_time(df_combined, df_enriched)` | Sorts targets chronologically by first-detection MJD — sets scheduler priority. |
| `simulate_telescope_scheduling_fast(df_sorted)` | Greedy 1-hour slot scheduler. Iterates targets in priority order; for each target, finds the first available slot across all windows and all telescopes. Returns `(master_calendars, scheduled_lists)`. |
| `plot_delay_distribution(scheduled_lists, filter_column)` | Histogram of trigger-to-observation delay times per telescope. |
| `plot_comprehensive_efficiency(scheduled_lists, main_filter)` | ECDF of turnaround time by observatory and by filter level. |
| `plot_scheduling_dashboard(scheduled_lists, filter_name)` | 3-panel dashboard: Gantt chart, delay histogram, workload bar chart. |
| `plot_advanced_funnel(df_master_windows, df_enriched, scheduled_lists)` | Interactive constraint funnel explorer with dropdown to switch between filter levels and a final "Scheduled" view with diamond markers. |

---

### `core/observatories/observatory_night_starlart_sim.py`

Validation tools that compare the simulation output against the [Staralt](https://astro.ing.iac.es/staralt/) web tool.

| Function | Description |
|----------|-------------|
| `plot_staralt_clone(df_master_windows, observatory_name, target_ids, obs_date_str)` | Plots elevation vs. time for selected targets on a given night, using computed telemetry. |
| `plot_all_targets_staralt(df_master_windows, observatory_name, obs_date_str)` | Same as above for all targets visible on a given night. |
| `fetch_and_display_staralt_gif_direct(target_name, ra_str, dec_str, obs_date_str, obs_name, lon, lat, alt, tz)` | Fetches the official single-target Staralt GIF and saves to `data/staralt_gifs/`. |
| `fetch_and_display_staralt_gif_multi(targets, obs_date_str, obs_name, lon, lat, alt, tz)` | Same for multiple targets in one request. |
| `get_staralt_target_info(df_enriched, target_id, custom_date_str)` | Prints the Staralt-compatible coordinate string for a target. |
| `get_staralt_observatory_info(obs)` | Prints the Staralt-compatible site string for an observatory. |

GIFs are saved to `data/staralt_gifs/` with filenames encoding the observatory and date.

---

## Configuration: Adding a New Telescope

1. Add a weather profile dict in `core/config/settings.py`:
   ```python
   my_obs_weather = {1: 0.70, 2: 0.72, ..., 12: 0.68}
   ```

2. Instantiate an `Observatory`:
   ```python
   MY_OBS = Observatory(
       name="MY_OBS",
       lon_str="...", lat_str="...", elevation_m=1000, tz_str="...",
       limiting_mag=24, weather_profile=my_obs_weather,
       sun_horizon=-15*u.deg, airmass_horizon=30*u.deg,
       moon_base_dist=20.0, moon_max_dist=70.0,
       seeing_median=1.0, seeing_limit=2.0,
       min_obs_window=1
   )
   ```

3. Add it to `ALL_TELESCOPES`:
   ```python
   ALL_TELESCOPES = [MAST, SOXS, WILDS, MY_OBS]
   ```

4. Delete the cache files in `data/` and rerun.

---

## Output Files

| File | Description |
|------|-------------|
| `data/enriched_lightcurves.csv` | Input data + peak/trigger columns. Delete to re-enrich. |
| `data/master_windows.pkl` | Visibility + weather windows for all targets. Delete to recompute. |
| `data/staralt_gifs/*.gif` | Official Staralt elevation plots downloaded for validation. |
