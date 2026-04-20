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
        for _ in range(10):
            H, dP, W = [], [], []
            up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])

            for i in valid_idx:
                vec = sat_p[i] - x_temp[:3]
                d = np.linalg.norm(vec)
                el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087)

                H.append([-vec[0]/d, -vec[1]/d, -vec[2]/d, 1.0])
                dP.append(prs[i] - (d + x_temp[3] + 2.4/np.sin(el))) # Add Tropo delay
                W.append(np.sin(el)**2)

            H, dP, W = np.array(H), np.array(dP), np.diag(W)
            try:
                update = np.linalg.solve(H.T @ W @ H, H.T @ W @ dP)
                x_temp += update
                if np.linalg.norm(update[:3]) < 1e-3: break
            except: break

        # Check Residuals
        residuals = []
        up_vec = x_temp[:3] / np.linalg.norm(x_temp[:3])
        for i in valid_idx:
            vec = sat_p[i] - x_temp[:3]
            d = np.linalg.norm(vec)
            el = max(np.arcsin(np.dot(vec, up_vec) / d), 0.087)
            res = abs(prs[i] - (d + x_temp[3] + 2.4/np.sin(el)))
            residuals.append(res)

        # Reject bad satellites (over 100m error)
        max_res = max(residuals)
        if max_res > 100:
            valid_idx.pop(residuals.index(max_res))
        else:
            x = x_temp
            break 

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
    except: v = np.zeros(4)
    return x, v

def main():
    C = 299792458.0
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    print("Loading RINEX files...")
    obs_data = gr.load(r'rinex_files\gnss_log_2026_03_21_17_14_34.26o', use='G') 
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260800000_01D_MN.rnx', use='G')
    
    # Using strict EPSG codes: ECEF to WGS84 (Lon, Lat, Alt)
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    
    last_p, last_v, last_time = None, np.zeros(3), None
    results = []

    MAP = { 'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'], 'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'], 'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'], 'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'], 'Cuc': ['Cuc'], 'Cus': ['Cus'], 'Crc': ['Crc'], 'Crs': ['Crs'], 'Cic': ['Cic'], 'Cis': ['Cis'] }

    print("Solving accurate path...")
    for epoch_time in obs_data.time.values:
        obs_epoch = obs_data.sel(time=epoch_time)
        sat_p, sat_v, prs, dops = [], [], [], []
        
        # --- THE TIME BUG FIX ---
        # RINEX observation time is ALREADY GPS Time. Do not add 18 seconds!
        t_gps = pd.to_datetime(epoch_time)
        # Strip timezone data to prevent Python from shifting it based on your PC's clock
        if t_gps.tzinfo is not None: t_gps = t_gps.tz_localize(None) 
        
        dt = (t_gps - last_time).total_seconds() if last_time is not None else 1.0
        last_time = t_gps
        
        # Seconds since Jan 6 1980
        tow = (t_gps - pd.Timestamp('1980-01-06')).total_seconds() % 604800 

        for sv in obs_epoch.sv.values:
            pr = obs_epoch['C1C'].sel(sv=sv).values if 'C1C' in obs_epoch else np.nan
            dop = obs_epoch['D1C'].sel(sv=sv).values if 'D1C' in obs_epoch else np.nan
            snr = obs_epoch['S1C'].sel(sv=sv).values if 'S1C' in obs_epoch else 35.0
            
            # Lowered SNR threshold to 20 to ensure we don't drop too many good points
            if not np.isnan(pr) and not np.isnan(dop) and not np.isnan(snr) and snr >= 20.0 and sv in nav_data.sv:
                try:
                    nav_msg = nav_data.sel(sv=sv).dropna(dim='time', how='all').sel(time=epoch_time, method='nearest', tolerance=pd.Timedelta(hours=4))
                    n_d = {'pr': pr}
                    for k, opts in MAP.items():
                        for o in opts:
                            if o in nav_msg: n_d[k] = float(nav_msg[o].values); break
                    
                    s_p, s_v, s_c = calculate_satellite_data(n_d, tow - (pr/C))
                    sat_p.append(s_p); sat_v.append(s_v); prs.append(pr + (s_c * C)); dops.append(dop)
                except: continue
                    
        if len(prs) >= 4:
            guess = ISRAEL_CENTER if last_p is None else last_p
            p_raw, v_raw = solve_pv_with_raim(sat_p, sat_v, prs, dops, guess)
            
            if p_raw is not None:
                # Basic Kinematic Smoothing 
                if last_p is not None and np.linalg.norm(v_raw[:3]) < 40:
                    p_pred = last_p[:3] + (last_v * dt)
                    p_final = (0.2 * p_raw[:3]) + (0.8 * p_pred)
                else:
                    p_final = p_raw[:3]

                last_p = np.append(p_final, p_raw[3])
                last_v = v_raw[:3]
                
                lon, lat, alt = transformer.transform(p_final[0], p_final[1], p_final[2])
                results.append({'UTC': t_gps.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt, 'Vx': v_raw[0], 'Vy': v_raw[1], 'Vz': v_raw[2]})

    if results:
        df = pd.DataFrame(results)
        
        # Drop extreme outliers (Lat/Lon should be in Israel roughly)
        df = df[(df['Lat'] > 29) & (df['Lat'] < 34) & (df['Lon'] > 33) & (df['Lon'] < 36)]
        
        df['Lat'] = df['Lat'].rolling(3, min_periods=1, center=True).mean()
        df['Lon'] = df['Lon'].rolling(3, min_periods=1, center=True).mean()
        
        df.to_csv("gnss_final_accurate.csv", index=False)
        
        kml = simplekml.Kml()
        kml.newlinestring(name="Accurate Path", coords=[(r['Lon'], r['Lat'], r['Alt']) for i, r in df.iterrows()]).style.linestyle.color = 'ff00ffff'
        kml.save("gnss_final_accurate.kml")
        print(f"SUCCESS: Exported {len(df)} points.")
    else:
        print("No points computed.")

if __name__ == "__main__":
    main()