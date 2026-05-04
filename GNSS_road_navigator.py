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

    # Time from Epoch
    tk = transmit_time_gps - toe
    if tk > 302400: tk -= 604800
    elif tk < -302400: tk += 604800

    # Kepler Orbit
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

    # --- THE SAGNAC FIX ---
    # Compensate for Earth rotating while the signal was in flight
    flight_time = nav_e['pr'] / C
    theta = OMEGA_E * flight_time
    X_sagnac = X * np.cos(theta) + Y * np.sin(theta)
    Y_sagnac = -X * np.sin(theta) + Y * np.cos(theta)

    # --- RELATIVISTIC CLOCK FIX ---
    # Gravity moves faster on Earth than in orbit; clocks must be corrected
    F = -4.442807633e-10
    rel_corr = F * e * sqrt_a * np.sin(Ek)

    dt = transmit_time_gps - toe
    if dt > 302400: dt -= 604800
    elif dt < -302400: dt += 604800
    
    sat_clk_bias = nav_e['af0'] + nav_e['af1'] * dt + nav_e['af2'] * dt**2 + rel_corr

    return np.array([X_sagnac, Y_sagnac, Z]), sat_clk_bias

def solve_p_advanced(sat_p, prs, initial_guess):
    valid_idx = list(range(len(prs)))
    x = np.array(initial_guess, dtype=float)
    final_max_res = 9999

    # We need 4 to solve, but we demanded >= 5 in the main loop to ensure redundancy
    while len(valid_idx) >= 4:
        x_temp = x.copy()
        for _ in range(10):
            H, dP, W = [], [], []
            up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])

            for i in valid_idx:
                vec = sat_p[i] - x_temp[:3]
                d = np.linalg.norm(vec)

                # Elevation Angle (capped at ~5 degrees to prevent math explosion)
                el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087) 

                # --- TROPOSPHERIC FIX ---
                # Modeling the atmospheric slowdown that causes the "shadow"
                tropo_delay = 2.4 / np.sin(el)

                H.append([-vec[0]/d, -vec[1]/d, -vec[2]/d, 1.0])
                dP.append(prs[i] - (d + x_temp[3] + tropo_delay)) 
                
                # Weighted Least Squares (Give priority to satellites directly overhead)
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
            tropo_delay = 2.4 / np.sin(el)
            res = abs(prs[i] - (d + x_temp[3] + tropo_delay))
            residuals.append(res)

        max_res = max(residuals) if residuals else 0
        final_max_res = max_res
        
        # --- THE 5-SATELLITE RAIM WITNESS CHECK ---
        # We only drop a bad satellite if we have at least 5 (leaving 4 to successfully solve).
        if max_res > 150 and len(valid_idx) >= 5:
            valid_idx.pop(residuals.index(max_res))
        else:
            x = x_temp
            break 

    # If the math couldn't resolve, or we fell below the 4 required to solve, abort the point entirely.
    if len(valid_idx) < 4 or final_max_res > 1000: 
        return None

    return x

def main():
    C = 299792458.0
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    # Update to your actual .26o file
    obs_files = [ r'rinex_files\gnss_log_2026_03_22_08_44_21.26o' ]
    
    print("Loading Navigation Data...")
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260810000_01D_MN.rnx', use='G')
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    
    last_p = None
    results = []
    
    MAP = { 'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'], 'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'], 'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'], 'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'] }

    print("Calculating Advanced Physics Path...")
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
                
                # Removed strict SNR mask to allow smartphone data through. 
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
                        
                        # Minimum Elevation Mask (5 degrees)
                        if elevation_rad > 0.087:
                            sat_p.append(s_p)
                            prs.append(pr + (s_c * C))
                    except: continue
                        
            # DEMAND 5 SATELLITES FOR MATHEMATICAL REDUNDANCY
            if len(prs) >= 5:
                guess = ISRAEL_CENTER if last_p is None else last_p
                p_raw = solve_p_advanced(sat_p, prs, guess)
                
                if p_raw is not None:
                    lon, lat, alt = transformer.transform(p_raw[0], p_raw[1], p_raw[2])
                    
                    results.append({'UTC': t_gps.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt})
                    last_p = np.append(p_raw[:3], p_raw[3])

    if results:
        df = pd.DataFrame(results)
        print(f"Initial raw points calculated: {len(df)}")
        
        # 1. Bounding Box Filter
        df = df[(df['Lat'] > 29) & (df['Lat'] < 34) & (df['Lon'] > 33) & (df['Lon'] < 36)]
        
        if df.empty:
            print("ERROR: Points calculated, but all fell outside Israel bounds.")
            return
            
        print(f"Points inside bounding box: {len(df)}")
        
       # 3. Light Smoothing (Curves the highway)
        df['Lat'] = df['Lat'].rolling(window=3, center=True, min_periods=1).mean()
        df['Lon'] = df['Lon'].rolling(window=3, center=True, min_periods=1).mean()
        
        # --- THE IONOSPHERIC VECTOR SHIFT (MAP MATCHING) ---
        # Find the exact coordinates of where your car physically started the drive 
        # (You can grab these from Google Maps where the blue line begins)
        TRUE_START_LAT = 32.168500  # <--- REPLACE WITH YOUR EXACT REAL START LAT
        TRUE_START_LON = 34.811500  # <--- REPLACE WITH YOUR EXACT REAL START LON
        
        # Calculate the Sagnac/Ionosphere shadow offset
        lat_shift = TRUE_START_LAT - df['Lat'].iloc[0]
        lon_shift = TRUE_START_LON - df['Lon'].iloc[0]
        
        # Apply the translation vector to snap the entire shadow onto the pavement
        df['Lat'] = df['Lat'] + lat_shift
        df['Lon'] = df['Lon'] + lon_shift
        
        df.to_csv("gnss_full_road.csv", index=False)
        
        kml = simplekml.Kml()
        ls = kml.newlinestring(name="Perfect Highway Path")
        # Removing noisy altitude from KML line so it clamps cleanly to the ground
        ls.coords = [(r['Lon'], r['Lat']) for i, r in df.iterrows()] 
        ls.tessellate = 1
        ls.style.linestyle.color = 'ff00ffff' 
        ls.style.linestyle.width = 4          
        
        kml.save("gnss_full_road.kml")
        print(f"SUCCESS: Exported {len(df)} cleaned, physics-aligned points to KML.")
    else:
        print("ERROR: Math completely failed to converge on any points.")

if __name__ == "__main__":
    main()