import georinex as gr
import numpy as np
import pandas as pd
import simplekml
from pyproj import Transformer

def calculate_satellite_data(nav_e, transmit_time_gps):
    MU, OMEGA_E, C = 3.986005e14, 7.2921151467e-5, 299792458.0
    
    toe, af0, af1, af2 = nav_e['Toe'], nav_e['af0'], nav_e['af1'], nav_e['af2']
    sqrt_a, e, i0, o0, w, M0 = nav_e['sqrtA'], nav_e['Eccentricity'], nav_e['Io'], nav_e['Omega0'], nav_e['M0']
    dn, odot, idot = nav_e['DeltaN'], nav_e['OmegaDot'], nav_e['IDOT']
    Cuc, Cus, Crc, Crs, Cic, Cis = nav_e['Cuc'], nav_e['Cus'], nav_e['Crc'], nav_e['Crs'], nav_e['Cic'], nav_e['Cis']

    dt = transmit_time_gps - toe
    if dt > 302400: dt -= 604800
    elif dt < -302400: dt += 604800
    sat_clk_bias = af0 + af1 * dt + af2 * dt**2
    tk = transmit_time_gps - sat_clk_bias - toe

    a = sqrt_a**2
    n = np.sqrt(MU/a**3) + dn
    Ek = M0 + n * tk
    for _ in range(10):
        Ek_old = Ek
        Ek = M0 + n * tk + e * np.sin(Ek)
        if abs(Ek - Ek_old) < 1e-12: break
    
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

    theta = OMEGA_E * (nav_e['pr'] / C)
    pos_final = np.array([Xk*np.cos(theta) + Yk*np.sin(theta), -Xk*np.sin(theta) + Yk*np.cos(theta), Zk])
    
    return pos_final, np.zeros(3), sat_clk_bias # Stripped out the broken Doppler velocity math

def solve_p_with_raim(sat_p, prs, initial_guess):
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

                H.append([-vec[0]/d, -vec[1]/d, -vec[2]/d, 1.0])
                dP.append(prs[i] - (d + x_temp[3] + 2.4/np.sin(el))) 
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
            res = abs(prs[i] - (d + x_temp[3] + 2.4/np.sin(el)))
            residuals.append(res)

        max_res = max(residuals) if residuals else 0
        final_max_res = max_res
        
        # Basic RAIM: Drop worst satellite if error > 150m
        if max_res > 150 and len(valid_idx) > 4:
            valid_idx.pop(residuals.index(max_res))
        else:
            x = x_temp
            break 

    # If the error is massive, return None so it doesn't corrupt the file
    if len(valid_idx) < 4 or final_max_res > 1000: 
        return None

    return x

def main():
    C = 299792458.0
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    # Update to your actual file
    obs_files = [ r'rinex_files\gnss_log_2026_03_22_08_44_21.26o' ]
    
    print("Loading Navigation Data...")
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260810000_01D_MN.rnx', use='G')
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    
    last_p = None
    results = []
    MAP = { 'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'], 'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'], 'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'], 'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'], 'Cuc': ['Cuc'], 'Cus': ['Cus'], 'Crc': ['Crc'], 'Crs': ['Crs'], 'Cic': ['Cic'], 'Cis': ['Cis'] }

    print("Calculating Raw Path...")
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
                
                # Removed ALL SNR filters. Only 5 degree elevation mask.
                if not np.isnan(pr) and sv in nav_data.sv:
                    try:
                        nav_msg = nav_data.sel(sv=sv).dropna(dim='time', how='all').sel(time=epoch_time, method='nearest', tolerance=pd.Timedelta(hours=4))
                        n_d = {'pr': pr}
                        for k, opts in MAP.items():
                            for o in opts:
                                if o in nav_msg: n_d[k] = float(nav_msg[o].values); break
                        
                        s_p, _, s_c = calculate_satellite_data(n_d, tow - (pr/C))
                        
                        guess = ISRAEL_CENTER if last_p is None else last_p
                        vec = s_p - guess[:3]
                        d = np.linalg.norm(vec)
                        up_vec = guess[:3] / np.linalg.norm(guess[:3])
                        elevation_rad = max(np.arcsin(np.dot(vec, up_vec) / d), 0)
                        
                        if elevation_rad > 0.087:
                            sat_p.append(s_p)
                            prs.append(pr + (s_c * C))
                    except: continue
                        
            if len(prs) >= 4:
                guess = ISRAEL_CENTER if last_p is None else last_p
                
                # We only solve for position. Ignore velocity completely.
                p_raw = solve_p_with_raim(sat_p, prs, guess)
                
                if p_raw is not None:
                    lon, lat, alt = transformer.transform(p_raw[0], p_raw[1], p_raw[2])
                    
                    # Store EVERYTHING. Let Pandas sort out the garbage later.
                    results.append({'UTC': t_gps.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt})
                    last_p = np.append(p_raw[:3], p_raw[3])

    if results:
        df = pd.DataFrame(results)
        
        # 1. Broad Bounding Box (Only deletes points in Egypt/Ocean/Space)
        df = df[(df['Lat'] > 29) & (df['Lat'] < 34) & (df['Lon'] > 33) & (df['Lon'] < 36)]
        
        if df.empty:
            print("ERROR: Points calculated, but all fell outside Israel bounds. Check time formatting.")
            return
            
        # 2. THE MEDIAN FILTER (The ultimate spike killer)
        # Sorts a window of 11 seconds. The 5 worst spikes are literally deleted.
        df['Lat'] = df['Lat'].rolling(window=11, center=True, min_periods=1).median()
        df['Lon'] = df['Lon'].rolling(window=11, center=True, min_periods=1).median()
        df['Alt'] = df['Alt'].rolling(window=11, center=True, min_periods=1).median()
        
        # 3. Light smoothing to make the road curve naturally
        df['Lat'] = df['Lat'].rolling(window=5, center=True, min_periods=1).mean()
        df['Lon'] = df['Lon'].rolling(window=5, center=True, min_periods=1).mean()
        
        df.to_csv("gnss_full_road.csv", index=False)
        
        kml = simplekml.Kml()
        ls = kml.newlinestring(name="Median Filtered Highway Path")
        ls.coords = [(r['Lon'], r['Lat'], r['Alt']) for i, r in df.iterrows()]
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        ls.tessellate = 1
        ls.style.linestyle.color = 'ff00ffff' 
        ls.style.linestyle.width = 4          
        
        start_row = df.iloc[0]
        start_pin = kml.newpoint(name="START", coords=[(start_row['Lon'], start_row['Lat'], start_row['Alt'])])
        start_pin.style.iconstyle.color = 'ff00ff00'
        
        kml.save("gnss_full_road.kml")
        print(f"SUCCESS: Exported {len(df)} cleaned points to KML.")
    else:
        print("ERROR: All points were filtered out.")

if __name__ == "__main__":
    main()