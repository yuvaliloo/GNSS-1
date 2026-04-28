import georinex as gr
import numpy as np
import pandas as pd
import simplekml
from pyproj import Transformer

def calculate_satellite_data(nav_e, transmit_time_gps):
    MU, OMEGA_E, C = 3.986005e14, 7.2921151467e-5, 299792458.0
    
    toe, af0, af1, af2 = nav_e['Toe'], nav_e['af0'], nav_e['af1'], nav_e['af2']
    sqrt_a, e, i0, o0, w, M0 = nav_e['sqrtA'], nav_e['Eccentricity'], nav_e['Io'], nav_e['Omega0'], nav_e['omega'], nav_e['M0']
    dn, odot, idot = nav_e['DeltaN'], nav_e['OmegaDot'], nav_e['IDOT']
    Cuc, Cus, Crc, Crs, Cic, Cis = nav_e['Cuc'], nav_e['Cus'], nav_e['Crc'], nav_e['Crs'], nav_e['Cic'], nav_e['Cis']

    # 1. Clock Correction
    dt = transmit_time_gps - toe
    if dt > 302400: dt -= 604800
    elif dt < -302400: dt += 604800
    sat_clk_bias = af0 + af1 * dt + af2 * dt**2
    tk = transmit_time_gps - sat_clk_bias - toe

    # 2. Kepler Orbit Math
    a = sqrt_a**2
    n = np.sqrt(MU/a**3) + dn
    Ek = M0 + n * tk
    for _ in range(10):
        Ek_old = Ek
        Ek = M0 + n * tk + e * np.sin(Ek)
        if abs(Ek - Ek_old) < 1e-12: break
    
    # Relativistic Correction
    sat_clk_bias += (-2 * np.sqrt(MU) / (C**2)) * e * sqrt_a * np.sin(Ek)

    vk = np.arctan2(np.sqrt(1 - e**2) * np.sin(Ek), np.cos(Ek) - e)
    pk = vk + w
    uk = pk + Cus * np.sin(2*pk) + Cuc * np.cos(2*pk)
    rk = a * (1 - e * np.cos(Ek)) + Crs * np.sin(2*pk) + Crc * np.cos(2*pk)
    ik = i0 + idot * tk + Cis * np.sin(2*pk) + Cic * np.cos(2*pk)
    
    x_p, y_p = rk * np.cos(uk), rk * np.sin(uk)
    Ok = o0 + (odot - OMEGA_E) * tk - OMEGA_E * toe
    
    Xk = x_p * np.cos(Ok) - y_p * np.cos(ik) * np.sin(Ok)
    Yk = x_p * np.sin(Ok) + y_p * np.cos(ik) * np.cos(Ok)
    Zk = y_p * np.sin(ik)

    # 3. Sagnac Effect
    theta = OMEGA_E * (nav_e['pr'] / C)
    pos_final = np.array([Xk*np.cos(theta) + Yk*np.sin(theta), -Xk*np.sin(theta) + Yk*np.cos(theta), Zk])

    # 4. Velocity Math
    edot = n / (1 - e * np.cos(Ek))
    vdot = np.sqrt(1 - e**2) * edot / (1 - e * np.cos(Ek))
    ukdot = vdot * (1 + 2 * Cus * np.cos(2*pk) - 2 * Cuc * np.sin(2*pk))
    rkdot = a * e * np.sin(Ek) * edot + 2 * vdot * (Crs * np.cos(2*pk) - Crc * np.sin(2*pk))
    ikdot = idot + 2 * vdot * (Cis * np.cos(2*pk) - Cic * np.sin(2*pk))
    Okdot = odot - OMEGA_E

    x_p_dot = rkdot * np.cos(uk) - rk * np.sin(uk) * ukdot
    y_p_dot = rkdot * np.sin(uk) + rk * np.cos(uk) * ukdot

    V_x = x_p_dot * np.cos(Ok) - y_p_dot * np.cos(ik) * np.sin(Ok) + y_p * np.sin(ik) * ikdot * np.sin(Ok) - Xk * Okdot
    V_y = x_p_dot * np.sin(Ok) + y_p_dot * np.cos(ik) * np.cos(Ok) - y_p * np.sin(ik) * ikdot * np.cos(Ok) + Yk * Okdot
    V_z = y_p_dot * np.sin(ik) + rk * np.sin(uk) * np.cos(ik) * ikdot
    
    return pos_final, np.array([V_x, V_y, V_z]), sat_clk_bias

def solve_pv_with_raim(sat_p, sat_v, prs, dops, initial_guess, C=299792458.0, L1=1575.42e6):
    valid_idx = list(range(len(prs)))
    x = np.array(initial_guess, dtype=float)

    # RAIM Loop for Position
    while len(valid_idx) >= 4:
        x_temp = x.copy()
        
        # Weighted Least Squares (WLS)
        for _ in range(10):
            H, dP, W = [], [], []
            up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])

            for i in valid_idx:
                vec = sat_p[i] - x_temp[:3]
                d = np.linalg.norm(vec)
                # Cap minimum elevation at 5 degrees to prevent divide-by-zero
                el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087)

                H.append([-vec[0]/d, -vec[1]/d, -vec[2]/d, 1.0])
                dP.append(prs[i] - (d + x_temp[3] + 2.4/np.sin(el))) # Add Tropo delay
                W.append(np.sin(el)**2)

            H, dP, W = np.array(H), np.array(dP), np.diag(W)
            try:
                update = np.linalg.solve(H.T @ W @ H, H.T @ W @ dP)
                x_temp += update
                # If the update is tiny, we have converged on a position
                if np.linalg.norm(update[:3]) < 1e-3: break
            except: 
                break

        # Check Residuals (How far off is the math from the raw GPS data?)
        residuals = []
        up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])
        for i in valid_idx:
            vec = sat_p[i] - x_temp[:3]
            d = np.linalg.norm(vec)
            el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087)
            res = abs(prs[i] - (d + x_temp[3] + 2.4/np.sin(el)))
            residuals.append(res)

        # --- THE FIX: Relaxed Neighborhood Threshold ---
        max_res = max(residuals) if residuals else 0
        
        # 1. Increased threshold from 100m to 400m
        # 2. Safety lock: Do not pop if we are at exactly 4 satellites (starvation)
        if max_res > 400 and len(valid_idx) > 4:
            valid_idx.pop(residuals.index(max_res))
        else:
            x = x_temp
            break 

    # If we somehow fell below 4, the epoch is dead
    if len(valid_idx) < 4: return None, None

    # Velocity Solve
    H_v, dD = [], []
    for i in valid_idx:
        vec = sat_p[i] - x[:3]
        d = np.linalg.norm(vec)
        unit = vec / d
        range_rate_obs = -dops[i] * (C / L1)
        range_rate_exp = np.dot(unit, sat_v[i])
        dD.append(range_rate_obs - range_rate_exp)
        H_v.append([-unit[0], -unit[1], -unit[2], 1.0]) 
        
    try:
        v = np.linalg.solve(np.array(H_v).T @ np.array(H_v), np.array(H_v).T @ np.array(dD))
    except: 
        v = np.zeros(4)
        
    return x, v

def main():
    C = 299792458.0
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    # 1. Put BOTH of your observation files in a list
    obs_files = [
        r'rinex_files\gnss_log_2026_03_21_17_14_34.26o', 
        r'rinex_files\gnss_log_2026_03_21_17_17_57.26o'
    ]
    
    print("Loading Navigation Data...")
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260800000_01D_MN.rnx', use='G')
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    
    # These states will persist across BOTH files, connecting the path
    last_p, last_v, last_time = None, np.zeros(3), None
    results = []

    MAP = { 'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'], 'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'], 'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'], 'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'], 'Cuc': ['Cuc'], 'Cus': ['Cus'], 'Crc': ['Crc'], 'Crs': ['Crs'], 'Cic': ['Cic'], 'Cis': ['Cis'] }

    print("Solving Combined Path...")
    
    # 2. Loop through both files sequentially
    for file_path in obs_files:
        print(f"Processing {file_path}...")
        try:
            obs_data = gr.load(file_path, use='G')
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        for epoch_time in obs_data.time.values:
            obs_epoch = obs_data.sel(time=epoch_time)
            sat_p, sat_v, prs, dops = [], [], [], []
            
            t_gps = pd.to_datetime(epoch_time)
            if t_gps.tzinfo is not None: t_gps = t_gps.tz_localize(None) 
            
            dt = (t_gps - last_time).total_seconds() if last_time is not None else 1.0
            tow = (t_gps - pd.Timestamp('1980-01-06')).total_seconds() % 604800 

            for sv in obs_epoch.sv.values:
                pr = float(obs_epoch['C1C'].sel(sv=sv).values) if 'C1C' in obs_epoch else np.nan
                dop = float(obs_epoch['D1C'].sel(sv=sv).values) if 'D1C' in obs_epoch else np.nan
                
                if not np.isnan(pr) and sv in nav_data.sv:
                    try:
                        nav_msg = nav_data.sel(sv=sv).dropna(dim='time', how='all').sel(time=epoch_time, method='nearest', tolerance=pd.Timedelta(hours=4))
                        n_d = {'pr': pr}
                        for k, opts in MAP.items():
                            for o in opts:
                                if o in nav_msg: n_d[k] = float(nav_msg[o].values); break
                        
                        s_p, s_v, s_c = calculate_satellite_data(n_d, tow - (pr/C))
                        sat_p.append(s_p); sat_v.append(s_v); prs.append(pr + (s_c * C))
                        dops.append(dop if not np.isnan(dop) else 0.0)
                    except: continue
                        
            if len(prs) >= 4:
                guess = ISRAEL_CENTER if last_p is None else last_p
                p_raw, v_raw = solve_pv_with_raim(sat_p, sat_v, prs, dops, guess)
                
                if p_raw is not None:
                    if last_p is not None:
                        # If dt > 5, it means we hit the 84-second gap between files.
                        # We ignore velocity prediction and start fresh so it doesn't jump wildly.
                        if np.linalg.norm(v_raw[:3]) < 40 and dt < 5:
                            p_pred = last_p[:3] + (last_v * dt)
                            p_final = (0.4 * p_raw[:3]) + (0.6 * p_pred)
                        else:
                            p_final = p_raw[:3]
                    else:
                        p_final = p_raw[:3]

                    last_time = t_gps
                    last_p = np.append(p_final, p_raw[3])
                    last_v = v_raw[:3]
                    
                    lon, lat, alt = transformer.transform(p_final[0], p_final[1], p_final[2])
                    results.append({'UTC': t_gps.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt})

    if results:
        df = pd.DataFrame(results)
        df = df[(df['Lat'] > 29) & (df['Lat'] < 34) & (df['Lon'] > 33) & (df['Lon'] < 36)]
        
        # Smooth the combined path
        df['Lat'] = df['Lat'].rolling(5, min_periods=1, center=True).mean()
        df['Lon'] = df['Lon'].rolling(5, min_periods=1, center=True).mean()
        
        df.to_csv("gnss_full_neighborhood.csv", index=False)
        
        kml = simplekml.Kml()
        ls = kml.newlinestring(name="Full Neighborhood Path")
        ls.coords = [(r['Lon'], r['Lat'], r['Alt']) for i, r in df.iterrows()]
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        ls.tessellate = 1
        ls.style.linestyle.color = 'ff00ffff' 
        ls.style.linestyle.width = 4          
        
        start_row = df.iloc[0]
        start_pin = kml.newpoint(name="START", coords=[(start_row['Lon'], start_row['Lat'], start_row['Alt'])])
        start_pin.style.labelstyle.scale = 1.0
        start_pin.style.iconstyle.color = 'ff00ff00'
        
        kml.save("gnss_full_neighborhood.kml")
        print(f"SUCCESS: Exported {len(df)} points. Maximum yield achieved across BOTH files.")
    else:
        print("No points computed.")

if __name__ == "__main__":
    main()