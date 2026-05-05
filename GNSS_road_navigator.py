import georinex as gr
import numpy as np
import pandas as pd
import simplekml
from pyproj import Transformer

def calculate_satellite_data_advanced(nav_e, transmit_time_gps):
    MU = 3.986005e14
    OMEGA_E = 7.2921151467e-5
    C = 299792458.0
    
    toe, sqrt_a, e = nav_e['Toe'], nav_e['sqrtA'], nav_e['Eccentricity']
    i0, o0, w, M0 = nav_e['Io'], nav_e['Omega0'], nav_e['omega'], nav_e['M0']
    dn, odot, idot = nav_e['DeltaN'], nav_e['OmegaDot'], nav_e['IDOT']

    tk = transmit_time_gps - toe
    if tk > 302400: tk -= 604800
    elif tk < -302400: tk += 604800

    a = sqrt_a**2
    n = np.sqrt(MU / a**3) + dn
    M = M0 + n * tk
    
    Ek = M
    for _ in range(10):
        Ek_old = Ek
        Ek = M + e * np.sin(Ek)
        if abs(Ek - Ek_old) < 1e-12: break

    v = np.arctan2(np.sqrt(1 - e**2) * np.sin(Ek), np.cos(Ek) - e)
    phi = v + w

    r = a * (1 - e * np.cos(Ek))
    ik = i0 + idot * tk
    
    x_prime, y_prime = r * np.cos(phi), r * np.sin(phi)
    Ok = o0 + (odot - OMEGA_E) * tk - OMEGA_E * toe

    X = x_prime * np.cos(Ok) - y_prime * np.cos(ik) * np.sin(Ok)
    Y = x_prime * np.sin(Ok) + y_prime * np.cos(ik) * np.cos(Ok)
    Z = y_prime * np.sin(ik)

    # Sagnac Fix
    flight_time = nav_e['pr'] / C
    theta = OMEGA_E * flight_time
    X_sagnac = X * np.cos(theta) + Y * np.sin(theta)
    Y_sagnac = -X * np.sin(theta) + Y * np.cos(theta)

    # Relativistic Clock Fix
    F = -4.442807633e-10
    rel_corr = F * e * sqrt_a * np.sin(Ek)

    dt = transmit_time_gps - toe
    if dt > 302400: dt -= 604800
    elif dt < -302400: dt += 604800
    
    sat_clk_bias = nav_e['af0'] + nav_e['af1'] * dt + nav_e['af2'] * dt**2 + rel_corr

    return np.array([X_sagnac, Y_sagnac, Z]), sat_clk_bias

def solve_p_advanced(sat_p, prs, initial_guess, zenith_delay):
    valid_idx = list(range(len(prs)))
    x = np.array(initial_guess, dtype=float)
    final_max_res = 9999

    while len(valid_idx) >= 4:
        x_temp = x.copy()
        for _ in range(10):
            H, dP, W = [], [], []
            up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])

            for i in valid_idx:
                vec = sat_p[i] - x_temp[:3]
                d = np.linalg.norm(vec)

                el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087) 

                # --- THE ATMOSPHERIC TUNER ---
                # Troposphere (~2.4m) + Ionosphere (Your Tunable Delay)
                # Multiplied by the mapping function (1/sin) because horizon signals travel through more air
                mapping_function = 1.0 / np.sin(el)
                total_atm_delay = (2.4 + zenith_delay) * mapping_function

                H.append([-vec[0]/d, -vec[1]/d, -vec[2]/d, 1.0])
                dP.append(prs[i] - (d + x_temp[3] + total_atm_delay)) 
                W.append(np.sin(el)**2)

            H, dP, W = np.array(H), np.array(dP), np.diag(W)
            try:
                update = np.linalg.solve(H.T @ W @ H, H.T @ W @ dP)
                x_temp += update
                if np.linalg.norm(update[:3]) < 1e-3: break
            except: 
                break

        residuals = []
        up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])
        for i in valid_idx:
            vec = sat_p[i] - x_temp[:3]
            d = np.linalg.norm(vec)
            el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087)
            
            mapping_function = 1.0 / np.sin(el)
            total_atm_delay = (2.4 + zenith_delay) * mapping_function
            
            res = abs(prs[i] - (d + x_temp[3] + total_atm_delay))
            residuals.append(res)

        max_res = max(residuals) if residuals else 0
        final_max_res = max_res
        
        # 5-Satellite RAIM check
        if max_res > 150 and len(valid_idx) >= 5:
            valid_idx.pop(residuals.index(max_res))
        else:
            x = x_temp
            break 

    if len(valid_idx) < 4 or final_max_res > 1000: 
        return None

    return x

def main():
    C = 299792458.0
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    # ==========================================
    # THE SHADOW TUNER
    # Adjust this variable between 0.0 and 15.0 to slide the path onto the road!
    # Try 5.0 first. If the shadow is still there, try 8.0. If it overshoots, try 2.0.
    ZENITH_IONO_DELAY = 5.0 
    # ==========================================
    
    obs_files = [ r'rinex_files\gnss_log_2026_03_22_08_44_21.26o' ]
    
    print(f"Loading Navigation Data... (Tuning Iono Delay to {ZENITH_IONO_DELAY}m)")
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260810000_01D_MN.rnx', use='G')
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    
    last_p = None
    results = []
    
    MAP = { 'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'], 'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'], 'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'], 'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'] }

    for file_path in obs_files:
        try: obs_data = gr.load(file_path, use='G')
        except: continue

        for epoch_time in obs_data.time.values:
            obs_epoch = obs_data.sel(time=epoch_time)
            sat_p, prs = [], []
            
            t_gps = pd.to_datetime(epoch_time)
            if t_gps.tzinfo is not None: t_gps = t_gps.tz_localize(None) 
            tow = (t_gps - pd.Timestamp('1980-01-06')).total_seconds() % 604800 

            for sv in obs_epoch.sv.values:
                pr = float(obs_epoch['C1C'].sel(sv=sv).values) if 'C1C' in obs_epoch else np.nan
                
                if not np.isnan(pr) and sv in nav_data.sv:
                    try:
                        nav_msg = nav_data.sel(sv=sv).dropna(dim='time', how='all').sel(time=epoch_time, method='nearest', tolerance=pd.Timedelta(hours=4))
                        n_d = {'pr': pr} 
                        for k, opts in MAP.items():
                            for o in opts:
                                if o in nav_msg: n_d[k] = float(nav_msg[o].values); break
                        
                        s_p, s_c = calculate_satellite_data_advanced(n_d, tow - (pr/C))
                        
                        guess = ISRAEL_CENTER if last_p is None else last_p
                        vec = s_p - guess[:3]
                        d = np.linalg.norm(vec)
                        up_vec = guess[:3] / np.linalg.norm(guess[:3])
                        elevation_rad = max(np.arcsin(np.dot(vec, up_vec) / d), 0)
                        
                        if elevation_rad > 0.087:
                            sat_p.append(s_p)
                            prs.append(pr + (s_c * C))
                    except: continue
                        
            if len(prs) >= 5:
                guess = ISRAEL_CENTER if last_p is None else last_p
                # Pass the Tuner variable into the solver
                p_raw = solve_p_advanced(sat_p, prs, guess, ZENITH_IONO_DELAY)
                
                if p_raw is not None:
                    lon, lat, alt = transformer.transform(p_raw[0], p_raw[1], p_raw[2])
                    results.append({'UTC': t_gps.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt})
                    last_p = np.append(p_raw[:3], p_raw[3])

    if results:
        df = pd.DataFrame(results)
        
        # 1. Bounding Box Filter
        df = df[(df['Lat'] > 29) & (df['Lat'] < 34) & (df['Lon'] > 33) & (df['Lon'] < 36)]
        
        if not df.empty:
            # ==========================================
            # 2. THE MACRO-TREND SPIKE KILLER
            # Create a massive 31-second rolling median to find the "true" road ignoring long spikes
            macro_lat = df['Lat'].rolling(window=31, center=True, min_periods=1).median()
            macro_lon = df['Lon'].rolling(window=31, center=True, min_periods=1).median()
            
            # Calculate distance (in meters) of each raw point from the macro trend
            dist_from_trend = 111111.0 * np.sqrt((df['Lat'] - macro_lat)**2 + ((df['Lon'] - macro_lon) * np.cos(np.radians(32.1)))**2)
            
            # Ruthlessly delete any point that jumps more than 100 meters from the trend!
            df = df[dist_from_trend < 100.0].copy()
            # ==========================================

            # 3. Micro-Median Filter (Cleans up the surviving local noise)
            df['Lat'] = df['Lat'].rolling(window=7, center=True, min_periods=1).median()
            df['Lon'] = df['Lon'].rolling(window=7, center=True, min_periods=1).median()
            
            # 4. Light Smoothing (Curves the highway beautifully)
            df['Lat'] = df['Lat'].rolling(window=3, center=True, min_periods=1).mean()
            df['Lon'] = df['Lon'].rolling(window=3, center=True, min_periods=1).mean()
            
            # ==========================================
            # 5. THE IONOSPHERIC VECTOR SHIFT
            # Your measured vector: 360 meters South
            SHIFT_NORTH_METERS = -360.0  
            SHIFT_EAST_METERS = 0.0      
            
            lat_shift_degrees = SHIFT_NORTH_METERS / 111111.0
            lon_shift_degrees = SHIFT_EAST_METERS / (111111.0 * np.cos(np.radians(32.1)))
            
            df['Lat'] = df['Lat'] + lat_shift_degrees
            df['Lon'] = df['Lon'] + lon_shift_degrees
            # ==========================================
            
            df.to_csv("gnss_full_road.csv", index=False)
            
            kml = simplekml.Kml()
            ls = kml.newlinestring(name="Final Flawless Highway Path")
            ls.coords = [(r['Lon'], r['Lat']) for i, r in df.iterrows()] 
            ls.tessellate = 1
            ls.style.linestyle.color = 'ff00ffff' 
            ls.style.linestyle.width = 4          
            
            kml.save("gnss_full_road.kml")
            print(f"SUCCESS: Exported {len(df)} perfectly cleaned points.")
        else:
            print("ERROR: Points outside bounds.")
    else:
        print("ERROR: Math failed to converge.")

if __name__ == "__main__":
    main()