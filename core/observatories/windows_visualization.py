import plotly.graph_objects as go
import pandas as pd
from astropy.time import Time

class ScheduleVisualizer:
    @staticmethod
    def mjd_to_dt(mjd_val):
        if pd.isna(mjd_val): return None
        return Time(mjd_val, format='mjd').datetime

    @classmethod
    def plot_funnel(cls, df_master_windows, df_enriched):
        print("🎨 Generating Constraint Funnel Explorer...")
        
        # 1. SETUP
        funnel_levels = [
            ('Windows_Mag_Only', '1. Mag Only', False),
            ('Windows_Mag_Sun', '2. Mag + Sun', False),
            ('Windows_Mag_Sun_Airmass', '3. + Airmass', False),
            ('mask_mag_sun_airmass_moon', '4. + Moon (Raw Geo)', False),
            ('mask_mag_sun_airmass_moon_duration', '5. Duration Filtered', False), 
            ('Windows_Final_Weather_Simulated', '6. Final (+ Weather)', True)      
        ]

        unique_obs = df_master_windows['Observatory'].unique()
        colors = ['darkgreen', 'darkorange', 'indigo', 'firebrick', 'teal']
        obs_configs = {}

        offset_step = 0.4 / max(1, len(unique_obs) - 1) if len(unique_obs) > 1 else 0
        start_offset = 0.2 if len(unique_obs) > 1 else 0

        for i, obs in enumerate(unique_obs):
            obs_configs[obs] = {'color': colors[i % len(colors)], 'offset': start_offset - (i * offset_step)}

        # 2. LIFESPANS & CHRONOLOGICAL ORDERING
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

        # 3. BUILD PLOTLY FIGURE
        fig = go.Figure()

        for level_col, level_name, is_visible in funnel_levels:
            x_good, y_good, txt_good = [], [], []
            x_bad, y_bad, txt_bad = [], [], []
            
            for tid in sorted_targets:
                min_mjd, max_mjd = lifespans[tid]
                if min_mjd is None: continue 
                    
                y_idx = target_y_map[tid]
                start_dt, end_dt = cls.mjd_to_dt(min_mjd), cls.mjd_to_dt(max_mjd)
                dur = max_mjd - min_mjd
                
                target_data = df_master_windows[df_master_windows['Target_ID'] == tid]
                total_windows = sum(target_data[level_col].apply(len))
                
                if total_windows == 0:
                    x_bad.extend([start_dt, end_dt, None])
                    y_bad.extend([y_idx, y_idx, None])
                    txt_bad.extend([f"<b>{tid}</b><br>FAILED at {level_name}", f"<b>{tid}</b><br>FAILED", None])
                else:
                    x_good.extend([start_dt, end_dt, None])
                    y_good.extend([y_idx, y_idx, None])
                    txt_good.extend([f"<b>{tid}</b><br>Visible Span: {dur:.1f}d", f"<b>{tid}</b><br>Visible Span: {dur:.1f}d", None])
                    
            fig.add_trace(go.Scatter(x=x_good, y=y_good, mode='lines', line=dict(color='lightsteelblue', width=2), text=txt_good, hoverinfo='text', name=f'Lifespan ({level_name})', visible=is_visible))
            fig.add_trace(go.Scatter(x=x_bad, y=y_bad, mode='lines', line=dict(color='red', width=4), text=txt_bad, hoverinfo='text', name=f'FAILED ({level_name})', visible=is_visible))

            for obs in unique_obs:
                obs_data = df_master_windows[df_master_windows['Observatory'] == obs]
                color, offset = obs_configs[obs]['color'], obs_configs[obs]['offset']
                
                x_vals, y_vals, txt_vals = [], [], []
                
                for _, row in obs_data.iterrows():
                    y_idx = target_y_map[row['Target_ID']]
                    for win in row[level_col]:
                        start_mjd, end_mjd, avg_mag = win
                        duration_mins = (end_mjd - start_mjd) * 24 * 60
                        
                        x_vals.extend([cls.mjd_to_dt(start_mjd), cls.mjd_to_dt(end_mjd), None])
                        y_vals.extend([y_idx + offset, y_idx + offset, None])
                        hover_html = f"<b>{row['Target_ID']}</b><br>Obs: <b>{obs}</b><br>Duration: {duration_mins:.1f} mins<br>Avg Mag: {avg_mag:.2f}"
                        txt_vals.extend([hover_html, hover_html, None])
                        
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=color, width=5), text=txt_vals, hoverinfo='text', name=f'{obs} ({level_name})', visible=is_visible))

        # 4. BUTTONS & LAYOUT
        traces_per_level = 2 + len(unique_obs) 
        def get_visibility(active_level_idx):
            vis = [False] * (len(funnel_levels) * traces_per_level)
            start_idx = active_level_idx * traces_per_level
            for i in range(traces_per_level):
                vis[start_idx + i] = True
            return vis

        buttons = [
            dict(label=level_name, method="update", args=[{"visible": get_visibility(i)}]) 
            for i, (_, level_name, _) in enumerate(funnel_levels)
        ]

        fig.update_layout(
            margin=dict(t=120, b=40, l=40, r=40),
            title=dict(text='<b>Global Observation Schedule: Constraint Funnel Explorer</b>', y=0.98, x=0.0),
            updatemenus=[dict(type="dropdown", direction="down", x=0.0, y=1.12, xanchor="left", yanchor="top", showactive=True, buttons=buttons)],
            xaxis_title='Date', 
            yaxis_title='Target Index',
            hovermode='closest', plot_bgcolor='white', height=800,
            xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat="%b %Y"), 
            yaxis=dict(showgrid=False) 
        )
        
        fig.show()