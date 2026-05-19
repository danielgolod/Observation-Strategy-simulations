import io
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
import re
from IPython.display import Image, display

STARALT_GIF_DIR = os.path.join("data", "staralt_gifs")

def fetch_and_display_staralt_gif_direct(target_name, ra_str, dec_str, obs_date_str,
                                         obs_name, lon, lat, alt, tz):
    """
    Single-target Staralt GIF fetch. All site parameters must be passed explicitly.
    """
    url = "https://astro.ing.iac.es/staralt/index.php"
    year, month, day = obs_date_str.split('-')
    clean_name = str(target_name).replace(' ', '_').replace('.', '_')
    coord_string = f"{lon} {lat} {alt} {tz}"

    multipart_payload = {
        'action':            (None, 'showImage'),
        'form[mode]':        (None, '1'),
        'form[day]':         (None, str(int(day)).zfill(2)),
        'form[month]':       (None, str(int(month)).zfill(2)),
        'form[year]':        (None, str(int(year))),
        'form[observatory]': (None, 'user'),
        'form[obs_name]':    (None, str(obs_name)),
        'form[sitecoord]':   (None, coord_string),
        'form[coordlist]':   (None, f"{clean_name} {ra_str} {dec_str}"),
        'form[minangle]':    (None, '30'),
        'form[paramdist]':   (None, '2'),
        'form[format]':      (None, 'gif'),
        'submit':            (None, 'Retrieve'),
    }

    os.makedirs(STARALT_GIF_DIR, exist_ok=True)
    filepath = os.path.join(STARALT_GIF_DIR, f"official_baseline_{clean_name}_{obs_name}.gif")

    print(f"Downloading official GIF for {clean_name} at {obs_name}...")
    try:
        response = requests.post(url, files=multipart_payload, timeout=15)
        response.raise_for_status()
        if response.content.startswith(b'GIF8'):
            with open(filepath, "wb") as f:
                f.write(response.content)
            display(Image(filename=filepath))
        else:
            print("FAILED: The server did not return a GIF.")
    except Exception as e:
        print(f"Connection failed: {e}")

def fetch_and_display_staralt_gif_multi(targets, obs_date_str, obs_name, lon, lat, alt, tz):
    """
    Multi-target Staralt GIF fetch. Sends all targets in a single request.
    targets: list of (target_name, ra_str, dec_str) tuples
    """
    url = "https://astro.ing.iac.es/staralt/index.php"
    year, month, day = obs_date_str.split('-')
    coord_string = f"{lon} {lat} {alt} {tz}"

    coord_lines = []
    for name, ra_str, dec_str in targets:
        clean_name = str(name).replace(' ', '_').replace('.', '_')
        coord_lines.append(f"{clean_name} {ra_str} {dec_str}")
    coordlist = "\n".join(coord_lines)

    multipart_payload = {
        'action':            (None, 'showImage'),
        'form[mode]':        (None, '1'),
        'form[day]':         (None, str(int(day)).zfill(2)),
        'form[month]':       (None, str(int(month)).zfill(2)),
        'form[year]':        (None, str(int(year))),
        'form[observatory]': (None, 'user'),
        'form[obs_name]':    (None, str(obs_name)),
        'form[sitecoord]':   (None, coord_string),
        'form[coordlist]':   (None, coordlist),
        'form[minangle]':    (None, '30'),
        'form[paramdist]':   (None, '2'),
        'form[format]':      (None, 'gif'),
        'submit':            (None, 'Retrieve'),
    }

    os.makedirs(STARALT_GIF_DIR, exist_ok=True)
    filepath = os.path.join(STARALT_GIF_DIR, f"official_baseline_multi_{obs_name}_{obs_date_str}.gif")

    print(f"Downloading official GIF for {len(targets)} targets at {obs_name}...")
    try:
        response = requests.post(url, files=multipart_payload, timeout=30)
        response.raise_for_status()
        if response.content.startswith(b'GIF8'):
            with open(filepath, "wb") as f:
                f.write(response.content)
            display(Image(filename=filepath))
        else:
            print("FAILED: The server did not return a GIF.")
    except Exception as e:
        print(f"Connection failed: {e}")


def plot_all_targets_with_official_overlay(df_master_windows, df_enriched, observatory_name, target_date_str):
    """
    Plots the pipeline's elevation arcs (thick solid lines) and automatically
    downloads/overlays the official Staralt data (thin dashed lines) for every target.
    """
    print(f"Plotting and Validating ALL targets for {observatory_name} on {target_date_str}...")
    
    df_obs = df_master_windows[df_master_windows['Observatory'] == observatory_name]
    
    if df_obs.empty:
        print("Warning: No data found for this observatory.")
        return

    window_start_time = Time(f"{target_date_str} 12:00:00")
    window_start_dt = window_start_time.datetime
    window_end_dt = (window_start_time + 1.0 * u.day).datetime
    window_start_mjd = window_start_time.mjd
    window_end_mjd = window_start_mjd + 1.0

    fig = go.Figure()
    moon_plotted = False
    target_count = 0
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for _, row in df_obs.iterrows():
        telemetry = row.get('Telemetry', {})
        if not isinstance(telemetry, dict) or 'Time_MJD' not in telemetry:
            continue
            
        mjd_array = np.array(telemetry['Time_MJD'])
        night_mask = (mjd_array >= window_start_mjd) & (mjd_array <= window_end_mjd)
        
        if not any(night_mask):
            continue 
            
        plot_mjds = mjd_array[night_mask]
        plot_times = Time(plot_mjds, format='mjd').datetime
        plot_elevs = np.array(telemetry['Target_Elevation'])[night_mask].astype(float)
        plot_moon_elevs = np.array(telemetry['Moon_Elevation'])[night_mask].astype(float)
        
        # Safely extract Sun Elevation (Defaults to NaN if running on old databases)
        raw_sun = telemetry.get('Sun_Elevation', [np.nan] * len(mjd_array))
        plot_sun_elevs = np.array(raw_sun)[night_mask].astype(float)
        
        # --- GAP BREAKER ---
        gap_indices = np.where(np.diff(plot_mjds) > 0.03)[0] + 1
        if len(gap_indices) > 0:
            dummy_times = plot_times[gap_indices - 1]
            plot_times = np.insert(plot_times, gap_indices, dummy_times)
            plot_elevs = np.insert(plot_elevs, gap_indices, np.nan)
            plot_moon_elevs = np.insert(plot_moon_elevs, gap_indices, np.nan)
            plot_sun_elevs = np.insert(plot_sun_elevs, gap_indices, np.nan) # Added Sun
        
        target_name = row['Target_ID']
        c_index = target_count % len(colors)
        line_color = colors[c_index]
        target_count += 1
        
        # 1. Plot Your Pipeline (Thick Solid)
        fig.add_trace(go.Scatter(
            x=plot_times, y=plot_elevs, mode='lines', 
            name=f"{target_name} (Pipeline)",
            line=dict(width=6, color=line_color), opacity=0.4 
        ))
        
        # 2. Fetch and Plot Staralt (Thin Dashed)
        target_data = df_enriched[df_enriched['Target_ID'] == target_name]
        if not target_data.empty:
            t_row = target_data.iloc[0]
            coords = SkyCoord(ra=t_row['RA (deg)'] * u.deg, dec=t_row['Dec (deg)'] * u.deg)
            ra_str = coords.ra.to_string(unit=u.hour, sep=' ', precision=1)
            dec_str = coords.dec.to_string(unit=u.degree, sep=' ', precision=1, alwayssign=True)
            
            print(f"  -> Scraping official baseline for {target_name}...")
            
            try:
                df_official = fetch_staralt_ascii(target_name, ra_str, dec_str, target_date_str)
                time.sleep(1.5) 
                
                if df_official is not None:
                    staralt_times = []
                    for dec_time in df_official['UT_Time']:
                        hour = int(dec_time)
                        minute = int(round((dec_time - hour) * 60))
                        if minute == 60: hour += 1; minute = 0
                        
                        if 12 <= hour < 24: st_time = window_start_dt.replace(hour=hour, minute=minute, second=0)
                        else: st_time = window_end_dt.replace(hour=hour, minute=minute, second=0)
                        staralt_times.append(st_time)

                    fig.add_trace(go.Scatter(
                        x=staralt_times, y=df_official['Elevation'], mode='lines', 
                        name=f"{target_name} (Official)",
                        line=dict(color=line_color, width=2, dash='dash'), hoverinfo='skip'
                    ))
            except NameError:
                print("  -> fetch_staralt_ascii not defined. Skipping overlay.")

        if not moon_plotted:
            fig.add_trace(go.Scatter(
                x=plot_times, y=plot_moon_elevs, mode='lines', 
                name='Moon', line=dict(color='black', width=3, dash='dot')
            ))
            # Plot the Sun if it exists in the telemetry
            if 'Sun_Elevation' in telemetry:
                fig.add_trace(go.Scatter(
                    x=plot_times, y=plot_sun_elevs, mode='lines', 
                    name='Sun', line=dict(color='orange', width=3, dash='dot')
                ))
            moon_plotted = True

    if target_count == 0:
        print("No telemetry data found for this night.")
        return

    # Add Horizontal Lines
    fig.add_hline(y=30, line_dash="dash", line_color="red", line_width=2)
    fig.add_hline(y=-18, line_dash="dot", line_color="purple", line_width=2,
                  annotation_text="-18 deg (Astronomical Twilight)", annotation_position="bottom left")

    fig.update_layout(
        title=f"<b>Pipeline vs. Staralt Validation: {observatory_name}</b><br>Night starting: {target_date_str}",
        xaxis_title="Time (UTC)", yaxis_title="Elevation (Degrees)",
        yaxis=dict(range=[-25, 90], dtick=10, showgrid=True, gridcolor='lightgray'), # Expanded Range
        xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat="%H:%M\n%b %d"),
        plot_bgcolor='white', height=800, hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
    )
    fig.show()


def plot_all_targets_staralt(df_master_windows, observatory_name, target_date_str,
                             tz_offset=0, elevation_threshold=30):
    """
    Plots ALL targets on a given night for the given observatory.
    Window is defined in LOCAL time: local noon of target_date -> local noon next day.
    """
    print(f"Plotting ALL targets for {observatory_name} on {target_date_str} (tz_offset={tz_offset:+d}h)...")

    df_obs = df_master_windows[df_master_windows['Observatory'] == observatory_name]
    if df_obs.empty:
        print("Warning: No data found for this observatory.")
        return

    # --- Time Setup (LOCAL noon -> next LOCAL noon, expressed in UT MJD) ---
    local_noon_as_ut_mjd = Time(f"{target_date_str} 12:00:00").mjd - (tz_offset / 24.0)
    window_start_mjd = local_noon_as_ut_mjd
    window_end_mjd   = window_start_mjd + 1.0

    fig = go.Figure()
    moon_plotted = False
    target_count = 0

    for _, row in df_obs.iterrows():
        telemetry = row.get('Telemetry', {})
        if not isinstance(telemetry, dict) or 'Time_MJD' not in telemetry:
            continue

        mjd_array = np.array(telemetry['Time_MJD'])
        night_mask = (mjd_array >= window_start_mjd) & (mjd_array <= window_end_mjd)
        if not any(night_mask):
            continue

        # Shift UT datetimes to local solar time for the x-axis.
        plot_times = np.array(
            [Time(m, format='mjd').datetime + pd.Timedelta(hours=tz_offset) for m in mjd_array[night_mask]],
            dtype=object
        )
        plot_elevs      = np.array(telemetry['Target_Elevation'])[night_mask].astype(float)
        plot_moon_elevs = np.array(telemetry['Moon_Elevation'])[night_mask].astype(float)
        raw_sun = np.array(telemetry.get('Sun_Elevation', [np.nan] * len(mjd_array)))[night_mask].astype(float)

        # --- GAP BREAKER ---
        gap_indices = np.where(np.diff(mjd_array[night_mask]) > 0.03)[0] + 1
        if len(gap_indices) > 0:
            plot_times      = np.insert(plot_times,      gap_indices, plot_times[gap_indices - 1])
            plot_elevs      = np.insert(plot_elevs,      gap_indices, np.nan)
            plot_moon_elevs = np.insert(plot_moon_elevs, gap_indices, np.nan)
            raw_sun         = np.insert(raw_sun,         gap_indices, np.nan)

        target_name = row['Target_ID']
        target_count += 1

        fig.add_trace(go.Scatter(
            x=plot_times, y=plot_elevs, mode='lines',
            name=target_name, line=dict(width=2), opacity=0.7
        ))

        if not moon_plotted:
            # Clip moon below horizon so the trace doesn't drag into negative range
            moon_for_plot = np.where(plot_moon_elevs < 0, np.nan, plot_moon_elevs)
            fig.add_trace(go.Scatter(
                x=plot_times, y=moon_for_plot, mode='lines',
                name='Moon', line=dict(color='gray', width=4, dash='dash'),
                connectgaps=False
            ))

            # Vertical sunset/sunrise + astronomical-twilight lines
            if len(raw_sun) > 0 and not np.isnan(raw_sun).all():
                cross_0 = np.where(np.diff(np.sign(raw_sun)))[0]
                for idx in cross_0:
                    if idx < len(plot_times):
                        fig.add_vline(x=plot_times[idx], line_width=1.5,
                                      line_dash="solid", line_color="black", opacity=0.5)
                cross_18 = np.where(np.diff(np.sign(raw_sun + 18)))[0]
                for idx in cross_18:
                    if idx < len(plot_times):
                        fig.add_vline(x=plot_times[idx], line_width=1.5,
                                      line_dash="dash", line_color="black", opacity=0.5)
            moon_plotted = True

    if target_count == 0:
        print("Zero targets found with telemetry on this night.")
        return

    print(f"Success! Plotted all {target_count} targets.")

    fig.add_hline(
        y=elevation_threshold, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"Horizon ({elevation_threshold} deg)", annotation_position="top left"
    )

    fig.update_layout(
        title=(f"<b>Staralt Offline Clone: {observatory_name}</b> "
               f"(night of {target_date_str}, tz {tz_offset:+d}h)<br>"
               f"Total Targets: {target_count}"),
        xaxis_title="Local Solar Time", yaxis_title="Elevation (Degrees)",
        yaxis=dict(range=[0, 90], dtick=10, showgrid=True, gridcolor='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat="%H:%M"),
        plot_bgcolor='white', height=800, hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
    )
    fig.show()


def plot_staralt_clone(df_master_windows, observatory_name, target_date_str, tz_offset=0, target_ids=None):
    """
    Creates a Staralt-style elevation plot.
    Sun arc is removed, vertical boundary lines for 0deg and -18deg are kept.
    Y-axis starts at the horizon (0 deg); -18 deg twilight horizontal line removed.
    Window is defined in LOCAL time: local noon of target_date -> local noon next day,
    so the observing night is centered in the plot (Staralt convention).
    """
    print(f"Generating Staralt Clone for {observatory_name} (tz_offset={tz_offset:+d}h)...")

    from datetime import datetime
    try:
        datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid TARGET_DATE '{target_date_str}': {e}")
        print("Use YYYY-MM-DD format and make sure the date exists (e.g. 2021 is not a leap year).")
        return

    df_obs = df_master_windows[df_master_windows['Observatory'] == observatory_name]
    if target_ids:
        df_obs = df_obs[df_obs['Target_ID'].isin(target_ids)]
    else:
        df_obs = df_obs.head(15)

    if df_obs.empty:
        return

    # --- Time Setup (LOCAL noon -> next LOCAL noon, expressed in UT MJD) ---
    # Local noon on target_date in UT  =  noon_UT(target_date) - tz_offset hours
    # (because local = UT + tz_offset  =>  UT = local - tz_offset)
    local_noon_as_ut_mjd = Time(f"{target_date_str} 12:00:00").mjd - (tz_offset / 24.0)
    window_start_mjd = local_noon_as_ut_mjd
    window_end_mjd   = window_start_mjd + 1.0  # 24h window

    fig = go.Figure()
    moon_plotted = False

    for _, row in df_obs.iterrows():
        telemetry = row.get('Telemetry', {})
        if not telemetry or 'Time_MJD' not in telemetry:
            continue

        mjd_array = np.array(telemetry['Time_MJD'])
        night_mask = (mjd_array >= window_start_mjd) & (mjd_array <= window_end_mjd)

        if not any(night_mask):
            t_min = Time(mjd_array.min(), format='mjd').datetime.strftime('%Y-%m-%d')
            t_max = Time(mjd_array.max(), format='mjd').datetime.strftime('%Y-%m-%d')
            print(f"  Skip {row['Target_ID']}: no telemetry on {target_date_str}. "
                  f"Telemetry covers {t_min} → {t_max}")
            continue

        # Shift UT datetimes to local solar time for the x-axis.
        # Build as a numpy object array so np.insert + fancy indexing work.
        plot_times = np.array(
            [Time(m, format='mjd').datetime + pd.Timedelta(hours=tz_offset) for m in mjd_array[night_mask]],
            dtype=object
        )
        plot_elevs      = np.array(telemetry['Target_Elevation'])[night_mask].astype(float)
        plot_moon_seps  = np.array(telemetry['Moon_Separation'])[night_mask].astype(float)
        plot_moon_elevs = np.array(telemetry['Moon_Elevation'])[night_mask].astype(float)
        # Sun data is used only to compute vertical sunset/twilight lines.
        raw_sun = np.array(telemetry.get('Sun_Elevation', [np.nan] * len(mjd_array)))[night_mask].astype(float)

        # --- GAP BREAKER ---
        gap_indices = np.where(np.diff(mjd_array[night_mask]) > 0.03)[0] + 1
        if len(gap_indices) > 0:
            plot_times      = np.insert(plot_times,      gap_indices, plot_times[gap_indices - 1])
            plot_elevs      = np.insert(plot_elevs,      gap_indices, np.nan)
            plot_moon_seps  = np.insert(plot_moon_seps,  gap_indices, np.nan)
            plot_moon_elevs = np.insert(plot_moon_elevs, gap_indices, np.nan)
            raw_sun         = np.insert(raw_sun,         gap_indices, np.nan)

        # Plot Target
        fig.add_trace(go.Scatter(
            x=plot_times, y=plot_elevs, mode='lines',
            name=row['Target_ID'], line=dict(width=2)
        ))

        # Moon-separation annotations along the target track
        annotate_mask = (np.arange(len(plot_times)) % 8 == 0) & ~np.isnan(plot_elevs)
        annotate_idx = np.where(annotate_mask)[0]
        fig.add_trace(go.Scatter(
            x=plot_times[annotate_idx],
            y=plot_elevs[annotate_idx],
            mode='text',
            text=[f"{sep:.0f} deg" for sep in plot_moon_seps[annotate_idx]],
            textposition="top center", textfont=dict(color="blue", size=10), showlegend=False
        ))

        if not moon_plotted:
            fig.add_trace(go.Scatter(
                x=plot_times, y=plot_moon_elevs, mode='lines',
                name='Moon', line=dict(color='gray', width=3, dash='dash')
            ))

            # VERTICAL BOUNDARY LINES (Calculation only, no Sun trace plotted)
            if len(raw_sun) > 0 and not np.isnan(raw_sun).all():
                # 0 deg (Sunset/Sunrise)
                cross_0 = np.where(np.diff(np.sign(raw_sun)))[0]
                for idx in cross_0:
                    if idx < len(plot_times):
                        fig.add_vline(x=plot_times[idx], line_width=1.5,
                                      line_dash="solid", line_color="black", opacity=0.5)
                # -18 deg (Astronomical Twilight)
                cross_18 = np.where(np.diff(np.sign(raw_sun + 18)))[0]
                for idx in cross_18:
                    if idx < len(plot_times):
                        fig.add_vline(x=plot_times[idx], line_width=1.5,
                                      line_dash="dash", line_color="black", opacity=0.5)
            moon_plotted = True

    # CLEAN HORIZON LINE (purple -18 deg twilight line removed)
    fig.add_hline(y=30, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="Horizon (30 deg)", annotation_position="top left")

    fig.update_layout(
        title=f"<b>Staralt Offline Clone: {observatory_name}</b> "
              f"(night of {target_date_str}, tz {tz_offset:+d}h)",
        xaxis_title="Local Solar Time", yaxis_title="Elevation (Degrees)",
        yaxis=dict(range=[0, 90], dtick=10, showgrid=True, gridcolor='lightgray'),
        xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat="%H:%M"),
        plot_bgcolor='white', height=700, hovermode="x unified"
    )
    fig.show()