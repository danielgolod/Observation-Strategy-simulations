import pandas as pd
import plotly.express as px
from astropy.time import Time
import plotly.graph_objects as go
from astropy.time import Time
import plotly.graph_objects as go
import numpy as np


def combine_target_windows(df_master):
    """
    Creates a copy of the dataframe, appends the observatory name to each window tuple,
    combines all observatories for a target, and sorts the windows chronologically.
    """
    df_copy = df_master.copy()
    
    window_columns = [
        'Windows_Mag_Only',
        'Windows_Mag_Sun',
        'Windows_Mag_Sun_Airmass',
        'mask_mag_sun_airmass_moon',
        'mask_mag_sun_airmass_moon_duration',
        'Windows_Final_Weather_Simulated'
    ]
    
    # 1. Add the observatory name to every tuple in every window list
    for col in window_columns:
        df_copy[col] = df_copy.apply(
            lambda row: [w + (row['Observatory'],) for w in row[col]] if isinstance(row[col], list) else row[col], 
            axis=1
        )
        
    # ---------------------------------------------------------
    # NEW: Custom function to combine and sort
    # ---------------------------------------------------------
    def flatten_and_sort(series_of_lists):
        # 1. sum(..., []) flattens the multiple lists into one big list
        combined_list = sum(series_of_lists, [])
        # 2. sorted() orders them chronologically by looking at the first element (x[0] = start MJD)
        return sorted(combined_list, key=lambda x: x[0])

    # 2. Group by Target_ID and apply our new custom flattening & sorting rule
    # Note: sort=False is kept here so the Target_ID order doesn't change alphabetically!
    df_combined = df_copy.groupby('Target_ID', sort=False)[window_columns].agg(flatten_and_sort).reset_index()
    
    return df_combined
    

def sort_combined_by_trigger_time(df_combined, df_enriched):
    """
    Takes the combined windows dataframe and sorts it chronologically
    based on the First_Observation_Time_MJD from df_enriched.
    """
    # 1. Extract just the Target_ID and Trigger Time from the enriched data
    # .drop_duplicates() ensures we just get one clean list of targets and times
    df_triggers = df_enriched[['Target_ID', 'First_Observation_Time_MJD']].drop_duplicates()
    
    # 2. Attach these trigger times directly to our combined dataframe
    # how='left' ensures we keep all our combined data safely intact
    df_merged = df_combined.merge(df_triggers, on='Target_ID', how='left')
    
    # 3. Sort the entire dataframe from earliest trigger time to latest
    df_sorted = df_merged.sort_values(by='First_Observation_Time_MJD', ascending=True)
    
    # 4. Reset the index so the row numbers start cleanly at 0, 1, 2...
    df_sorted = df_sorted.reset_index(drop=True)
    
    return df_sorted



def simulate_telescope_scheduling_fast(df_sorted):
   
    window_columns = [
        'Windows_Mag_Only', 'Windows_Mag_Sun', 'Windows_Mag_Sun_Airmass',
        'mask_mag_sun_airmass_moon', 'mask_mag_sun_airmass_moon_duration',
        'Windows_Final_Weather_Simulated'
    ]
    
    sub_lists = {col: [] for col in window_columns}
    
    # NEW: Create a master dictionary to hold the calendars for every filter!
    master_calendars = {col: {} for col in window_columns}
    
    duration = 1.0 / 24.0  # 1 hour in MJD
    
    for col in window_columns:
        
        telescope_calendars = {}

        for current_idx, row in df_sorted.iterrows():
            target_id = row['Target_ID']
            trigger_time = row['First_Observation_Time_MJD']
            windows = row[col]
            
            if not isinstance(windows, list) or len(windows) == 0:
                continue
            
            # Optional but highly recommended: Sort windows chronologically
            windows = sorted(windows, key=lambda x: x[0])
            
            for fw in windows:
                w_start, w_end, w_mag, obs = fw[0], fw[1], fw[2], fw[3]
                
                if obs not in telescope_calendars:
                    telescope_calendars[obs] = []
                    
                booked_slots = sorted(telescope_calendars[obs])
                
                candidate_start = w_start
                
                # IMPORTANT: Unpacking 4 items here since you are saving 4 items below!
                for b_start, b_end, b_target, b_trigger in booked_slots:
                    # Does our 1-hour slot safely fit BEFORE the higher-priority booking?
                    if candidate_start + duration <= b_start:
                        break
                    
                    # If we overlap with a higher-priority booking, we must wait!
                    if candidate_start < b_end:
                        candidate_start = b_end
                
                if candidate_start + duration <= w_end:
                    end_time = candidate_start + duration
                    delay_time = end_time - trigger_time
                    
                    telescope_calendars[obs].append((candidate_start, end_time, target_id, trigger_time))
                    sub_lists[col].append([target_id, candidate_start, end_time, delay_time, obs])
                    
                    break # Stop checking windows, target got its slot!
        
        # NEW: Save this specific filter's finished calendar into the master dictionary
        master_calendars[col] = telescope_calendars

    # CHANGED: Return the master_calendars instead of df_sorted
    return master_calendars, sub_lists

def plot_delay_distribution(scheduled_lists, filter_column='mask_mag_sun_airmass_moon'):
    """
    Plots a histogram of the time elapsed from the Supernova Trigger 
    until the END of the 1-hour telescope booking.
    """
    # 1. Grab the specific list of bookings you want to analyze
    bookings = scheduled_lists.get(filter_column, [])
    
    if not bookings:
        print(f"⚠️ No bookings found for the column: {filter_column}")
        return
        
    # 2. Extract the delay times (Index 3) and convert from Days to Hours
    # booking[3] is the delay_time we calculated previously
    delay_times_hours = [booking[3] * 24 for booking in bookings]
    
    # 3. Put it into a simple DataFrame for Plotly
    df_plot = pd.DataFrame({
        'Target_ID': [booking[0] for booking in bookings],
        'Delay_Hours': delay_times_hours,
        'Observatory': [booking[4] for booking in bookings]
    })
    
    # 4. Create an interactive Plotly Histogram
    fig = px.histogram(
        df_plot, 
        x='Delay_Hours', 
        color='Observatory', # Color-code by which telescope took the shot!
        nbins=40, # Adjust this to make the bars thicker or thinner
        title=f'<b>Time to Complete Spectrum Observation</b><br><sup>Filter Level: {filter_column}</sup>',
        labels={'Delay_Hours': 'Hours from Trigger to End of Observation'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # 5. Make it look clean
    fig.update_layout(
        yaxis_title="Number of Targets",
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Add a grid for readability
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    
    fig.show()

def plot_comprehensive_efficiency(scheduled_lists, main_filter='Windows_Final_Weather_Simulated'):
    """
    Generates two plots:
    1. Turnaround time broken down by Observatory + Total Network (for a specific filter).
    2. Turnaround time broken down by the different Filter Levels.
    """
    # ==========================================
    # PLOT 1: OBSERVATORIES & TOTAL NETWORK
    # ==========================================
    main_bookings = scheduled_lists.get(main_filter, [])
    
    if not main_bookings:
        print(f"⚠️ No bookings found for the column: {main_filter}")
        return
        
    # Build the base DataFrame
    df_main = pd.DataFrame({
        'Target_ID': [b[0] for b in main_bookings],
        'Turnaround_Time_Hours': [b[3] * 24 for b in main_bookings],
        'Observatory': [b[4] for b in main_bookings]
    })
    
    # Calculate Stats for the Total Network
    avg_time = df_main['Turnaround_Time_Hours'].mean()
    median_time = df_main['Turnaround_Time_Hours'].median()
    
    print(f"🔭 Total Targets Scheduled (Filter: {main_filter}): {len(df_main)}")
    print(f"⏱️ Average Time to Spectrum: {avg_time:.2f} hours")
    print(f"🎯 Median Time to Spectrum: {median_time:.2f} hours")
    print("📊 Generating Visualizations...")
    
    # THE TRICK: Duplicate the dataframe and label it "Total Network"
    df_total = df_main.copy()
    df_total['Observatory'] = 'Total Network'
    
    # Combine them so Plotly treats the "Total Network" as just another observatory category
    df_plot1 = pd.concat([df_main, df_total])
    
    fig1 = px.ecdf(
        df_plot1, 
        x="Turnaround_Time_Hours", 
        color="Observatory",
        title=f"<b>Cumulative Turnaround Time by Observatory</b><br><sup>Filter: {main_filter}</sup>",
        labels={"Turnaround_Time_Hours": "Hours to Spectrum", "probability": "Percentage of Targets Observed"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Style Plot 1 and make the 'Total Network' line pop out
    fig1.update_layout(yaxis=dict(tickformat=".0%"), template="plotly_dark", hovermode="x unified")
    fig1.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    fig1.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
    
    for trace in fig1.data:
        if trace.name == 'Total Network':
            trace.line.width = 4
            trace.line.dash = 'dash'
            trace.line.color = 'white'

    # Add the median line to Plot 1
    fig1.add_vline(x=median_time, line_dash="dash", line_color="white", 
                   annotation_text=f"Network Median: {median_time:.1f}h", annotation_position="bottom right")

    fig1.show()


    
def plot_scheduling_dashboard(scheduled_lists, filter_name='Windows_Final_Weather_Simulated'):
    """
    Generates a 3-part dashboard (Gantt, Histogram, Workload) 
    by unpacking the scheduled_lists dictionary.
    """
    print(f"📊 Generating Scheduling Dashboard for {filter_name}...")
    
    # 1. Safely extract the list from the dictionary
    final_bookings = scheduled_lists.get(filter_name, [])
    
    if not final_bookings:
        print("⚠️ No scheduled bookings found!")
        return

    # 2. Convert the raw list of lists into a Pandas DataFrame!
    # Your algorithm saves: [target_id, candidate_start, end_time, delay_time, obs]
    df_sched = pd.DataFrame(final_bookings, columns=['Target_ID', 'Start_MJD', 'End_MJD', 'Delay_Days', 'Observatory'])
    
    # Convert MJD to real Datetime objects and calculate hours
    df_sched['Start_Time'] = Time(df_sched['Start_MJD'].values, format='mjd').datetime
    df_sched['End_Time'] = Time(df_sched['End_MJD'].values, format='mjd').datetime
    df_sched['Turnaround_Time_Hours'] = df_sched['Delay_Days'] * 24

    # ==========================================
    # BOLD, RICH PALETTE
    # ==========================================
    obs_colors = {
        'MAST': '#005b96',   # Deep Ocean Blue
        'SOXS': '#d32d41',   # Strong Brick Red
        'WILDS': '#6a0572'   # Rich Plum Purple
    }

    # ==========================================
    # PLOT 1: THE OBSERVATORY GANTT CHART
    # ==========================================
    fig1 = px.timeline(
        df_sched, 
        x_start="Start_Time", 
        x_end="End_Time", 
        y="Observatory", 
        color="Observatory",
        hover_name="Target_ID",
        hover_data={"Start_Time": "|%b %d, %H:%M", "End_Time": False, "Observatory": False, "Turnaround_Time_Hours": ":.1f"},
        title="<b>Telescope Schedule (Gantt Chart)</b><br><sup>Actual utilization and idle time per observatory</sup>",
        color_discrete_map=obs_colors 
    )
    
    fig1.update_yaxes(autorange="reversed") 
    
    # Force opacity and thick borders so 1-hour blocks don't vanish
    fig1.update_traces(
        opacity=1.0, 
        marker_line_width=2, 
        marker_line_color='black'
    )

    fig1.update_layout(template="plotly_white", showlegend=False, height=400)
    fig1.show()

    # ==========================================
    # PLOT 2: RESPONSE TIME HISTOGRAM
    # ==========================================
    fig2 = px.histogram(
        df_sched, 
        x="Turnaround_Time_Hours", 
        color="Observatory",
        nbins=50,
        barmode="overlay", 
        opacity=0.75,
        title="<b>Response Time Distribution</b><br><sup>How long targets wait before being observed</sup>",
        labels={"Turnaround_Time_Hours": "Hours from Trigger to Observation"},
        color_discrete_map=obs_colors
    )
    fig2.update_layout(template="plotly_dark", height=400, hovermode="x unified")
    fig2.show()

    # ==========================================
    # PLOT 3: WORKLOAD / CAPACITY BAR CHART
    # ==========================================
    df_counts = df_sched['Observatory'].value_counts().reset_index()
    df_counts.columns = ['Observatory', 'Total_Targets_Observed']

    fig3 = px.bar(
        df_counts, 
        x="Observatory", 
        y="Total_Targets_Observed", 
        color="Observatory",
        text="Total_Targets_Observed",
        title="<b>Total Network Workload</b><br><sup>Number of targets successfully scheduled per telescope</sup>",
        color_discrete_map=obs_colors
    )
    fig3.update_traces(textposition='outside', textfont_size=14)
    fig3.update_layout(template="plotly_dark", height=400, showlegend=False)
    fig3.show()




def mjd_to_dt(mjd_val):
    if pd.isna(mjd_val): return None
    return Time(mjd_val, format='mjd').datetime

def plot_advanced_funnel(df_master_windows, df_enriched, scheduled_lists):
    """
    Generates the interactive Constraint Funnel Explorer, allowing users 
    to toggle between raw windows and final scheduled bookings (diamonds).
    """
    print("🎨 Generating Advanced Constraint Funnel Explorer...")
    
    # ==========================================
    # 1. SETUP THE FUNNEL & DYNAMIC OBSERVATORIES
    # ==========================================
    funnel_levels = [
        ('Windows_Mag_Only', '1. Mag Only', False),
        ('Windows_Mag_Sun', '2. Mag + Sun', False),
        ('Windows_Mag_Sun_Airmass', '3. + Airmass', False),
        ('mask_mag_sun_airmass_moon', '4. + Moon (Raw Geo)', False),
        ('mask_mag_sun_airmass_moon_duration', '5. Duration Filtered', False), 
        ('Windows_Final_Weather_Simulated', '6. Final (+ Weather)', False),
        ('SCHEDULED', '7. Final Scheduled Bookings', True)      
    ]

    unique_obs = df_master_windows['Observatory'].unique()
    colors = ['darkgreen', 'darkorange', 'indigo', 'firebrick', 'teal']
    obs_configs = {}

    offset_step = 0.4 / max(1, len(unique_obs) - 1) if len(unique_obs) > 1 else 0
    start_offset = 0.2 if len(unique_obs) > 1 else 0

    for i, obs in enumerate(unique_obs):
        obs_configs[obs] = {'color': colors[i % len(colors)], 'offset': start_offset - (i * offset_step)}

    # ==========================================
    # 2. CALCULATE LIFESPANS & ORDER TARGETS
    # ==========================================
    trigger_times = df_enriched.set_index('Target_ID')['First_Observation_Time_MJD'].to_dict()

    lifespans = {}
    for tid, group in df_master_windows.groupby('Target_ID'):
        all_mag_windows = sum(group['Windows_Mag_Only'].tolist(), [])
        if all_mag_windows:
            lifespans[tid] = (min([w[0] for w in all_mag_windows]), max([w[1] for w in all_mag_windows]))
        else:
            lifespans[tid] = (None, None)

    valid_targets = [tid for tid, span in lifespans.items() if span[0] is not None]
    valid_targets.sort(key=lambda tid: trigger_times.get(tid, float('inf')))
    failed_targets = [tid for tid, span in lifespans.items() if span[0] is None]

    sorted_targets = valid_targets + failed_targets
    target_y_map = {tid: idx for idx, tid in enumerate(sorted_targets)}

    # ==========================================
    # 3. BUILD THE PLOTLY FIGURE
    # ==========================================
    fig = go.Figure()

    # We will use trace_map to safely build the dropdown buttons later
    trace_map = [] 

    for lvl_idx, (level_col, level_name, is_visible) in enumerate(funnel_levels):
        x_good, y_good, txt_good = [], [], []
        x_bad, y_bad, txt_bad = [], [], []
        
        # A. Draw the Grey/Red Baselines (Target Lifespans)
        for tid in sorted_targets:
            min_mjd, max_mjd = lifespans[tid]
            if min_mjd is None: continue 
                
            y_idx = target_y_map[tid]
            start_dt, end_dt = mjd_to_dt(min_mjd), mjd_to_dt(max_mjd)
            dur = max_mjd - min_mjd
            
            if level_col != 'SCHEDULED':
                target_data = df_master_windows[df_master_windows['Target_ID'] == tid]
                total_windows = sum(target_data[level_col].apply(len))
            else:
                total_windows = 1 
            
            if total_windows == 0:
                x_bad.extend([start_dt, end_dt, None])
                y_bad.extend([y_idx, y_idx, None])
                txt_bad.extend([f"<b>{tid}</b><br>FAILED at {level_name}", f"<b>{tid}</b><br>FAILED", None])
            else:
                x_good.extend([start_dt, end_dt, None])
                y_good.extend([y_idx, y_idx, None])
                txt_good.extend([f"<b>{tid}</b><br>Visible Span: {dur:.1f}d", f"<b>{tid}</b><br>Visible Span: {dur:.1f}d", None])
                
        fig.add_trace(go.Scatter(x=x_good, y=y_good, mode='lines', line=dict(color='lightsteelblue', width=2), text=txt_good, hoverinfo='text', name=f'Lifespan ({level_name})', visible=is_visible))
        trace_map.append(lvl_idx)
        
        fig.add_trace(go.Scatter(x=x_bad, y=y_bad, mode='lines', line=dict(color='red', width=4), text=txt_bad, hoverinfo='text', name=f'FAILED ({level_name})', visible=is_visible))
        trace_map.append(lvl_idx)

        # B. Draw the Observatory Windows and/or Scheduled Bookings
        for obs in unique_obs:
            color = obs_configs[obs]['color']
            offset = obs_configs[obs]['offset']
            
            if level_col == 'SCHEDULED':
                # ----------------------------------------------------
                # 1. Plot the underlying valid windows (Semi-transparent background tracks)
                # ----------------------------------------------------
                underlying_col = 'Windows_Final_Weather_Simulated' # Change this if scheduling a different level
                obs_data = df_master_windows[df_master_windows['Observatory'] == obs]
                
                x_wins, y_wins, txt_wins = [], [], []
                for _, row in obs_data.iterrows():
                    y_idx = target_y_map[row['Target_ID']]
                    for win in row[underlying_col]:
                        start_mjd, end_mjd, avg_mag = win
                        x_wins.extend([mjd_to_dt(start_mjd), mjd_to_dt(end_mjd), None])
                        y_wins.extend([y_idx + offset, y_idx + offset, None])
                        duration_mins = (end_mjd - start_mjd) * 24 * 60
                        hover_html = f"<b>{row['Target_ID']}</b><br>Obs: <b>{obs}</b><br>Valid Window: {duration_mins:.1f} mins"
                        txt_wins.extend([hover_html, hover_html, None])
                
                fig.add_trace(go.Scatter(x=x_wins, y=y_wins, mode='lines', line=dict(color=color, width=4), opacity=0.4, text=txt_wins, hoverinfo='text', name=f'{obs} (Valid Window)', visible=is_visible))
                trace_map.append(lvl_idx)
                
                # ----------------------------------------------------
                # 2. Plot the scheduled 1-hour diamond ON TOP of the track
                # ----------------------------------------------------
                final_bookings = scheduled_lists.get(underlying_col, [])
                obs_bookings = [b for b in final_bookings if b[4] == obs]
                
                x_dia, y_dia, txt_dia = [], [], []
                for b in obs_bookings:
                    tid, start_mjd, end_mjd, delay, _ = b
                    y_idx = target_y_map[tid]
                    x_dia.append(mjd_to_dt(start_mjd))
                    y_dia.append(y_idx + offset)
                    hover_html = f"<b>{tid}</b><br>Obs: <b>{obs}</b><br>Delay from Trigger: {delay*24:.1f} hrs<br>Scheduled Time: {mjd_to_dt(start_mjd).strftime('%Y-%m-%d %H:%M')}"
                    txt_dia.append(hover_html)
                    
                fig.add_trace(go.Scatter(
                    x=x_dia, y=y_dia, mode='markers', 
                    marker=dict(color=color, size=10, symbol='diamond', line=dict(width=1.5, color='white')), 
                    text=txt_dia, hoverinfo='text', name=f'{obs} (Scheduled)', visible=is_visible
                ))
                trace_map.append(lvl_idx)
                
            else:
                # ----------------------------------------------------
                # STANDARD VIEW: Just plot the full availability windows
                # ----------------------------------------------------
                obs_data = df_master_windows[df_master_windows['Observatory'] == obs]
                x_vals, y_vals, txt_vals = [], [], []
                
                for _, row in obs_data.iterrows():
                    y_idx = target_y_map[row['Target_ID']]
                    for win in row[level_col]:
                        start_mjd, end_mjd, avg_mag = win
                        x_vals.extend([mjd_to_dt(start_mjd), mjd_to_dt(end_mjd), None])
                        y_vals.extend([y_idx + offset, y_idx + offset, None])
                        duration_mins = (end_mjd - start_mjd) * 24 * 60
                        hover_html = f"<b>{row['Target_ID']}</b><br>Obs: <b>{obs}</b><br>Duration: {duration_mins:.1f} mins<br>Avg Mag: {avg_mag:.2f}"
                        txt_vals.extend([hover_html, hover_html, None])
                        
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=color, width=5), text=txt_vals, hoverinfo='text', name=f'{obs} ({level_name})', visible=is_visible))
                trace_map.append(lvl_idx)

    # ==========================================
    # 4. CREATE THE BUTTONS AND FIX LAYOUT
    # ==========================================
    # Dynamically generate visibility boolean arrays based on the trace_map
    buttons = []
    for i, (_, level_name, _) in enumerate(funnel_levels):
        vis_array = [t == i for t in trace_map]
        buttons.append(dict(label=level_name, method="update", args=[{"visible": vis_array}]))

    fig.update_layout(
        margin=dict(t=120, b=40, l=40, r=40),
        title=dict(text='<b>Global Observation Schedule: Constraint Funnel Explorer</b>', y=0.98, x=0.0),
        updatemenus=[dict(
            type="dropdown", direction="down", x=0.0, y=1.12, xanchor="left", yanchor="top", showactive=True, buttons=buttons
        )],
        xaxis_title='Date', 
        yaxis_title='Target Index',
        hovermode='closest', plot_bgcolor='white', height=800,
        xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat="%b %Y"), 
        yaxis=dict(showgrid=False) 
    )

    fig.show()