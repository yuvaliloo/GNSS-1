import georinex as gr
import numpy as np
import pandas as pd
import simplekml
from pyproj import Transformer

def calculate_satellite_data(nav_e, transmit_time_gps):
    """Calculates Sat Position and Velocity."""
    MU, OMEGA_E, C = 3.986005e14, 7.2921151467e-5, 299792458.0
    
    # Orbit params
    toe, af0, af1, af2 = nav_e['Toe'], nav_e['af0'], nav_e['af1'], nav_e['af2']
    sqrt_a, e, i0, o0, w, M0 = nav_e['sqrtA'], nav_e['Eccentricity'], nav_e['Io'], nav_e['Omega0'], nav_e['omega'], nav_e['M0']
    dn, odot, idot = nav_e['DeltaN'], nav_e['OmegaDot'], nav_e['IDOT']
    Cuc, Cus, Crc, Crs, Cic, Cis = nav_e['Cuc'], nav_e['Cus'], nav_e['Crc'], nav_e['Crs'], nav_e['Cic'], nav_e['Cis']

    # 1. Clock & Time
    dt = transmit_time_gps - toe
    if dt > 302400: dt -= 604800
    elif dt < -302400: dt += 604800
    sat_clk_bias = af0 + af1 * dt + af2 * dt**2
    tk = transmit_time_gps - sat_clk_bias - toe

    # 2. Iterative Kepler
    a = sqrt_a**2
    n = np.sqrt(MU/a**3) + dn
    Mk = M0 + n * tk
    Ek = Mk
    for _ in range(10):
        Ek_old = Ek
        Ek = Mk + e * np.sin(Ek)
        if abs(Ek - Ek_old) < 1e-12: break
    
    # 3. Position
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

    # 4. Sagnac Effect
    theta = OMEGA_E * (nav_e['pr'] / C)
    pos_final = np.array([Xk*np.cos(theta) + Yk*np.sin(theta), -Xk*np.sin(theta) + Yk*np.cos(theta), Zk])

    # 5. Satellite Velocity (Simplified for Assignment)
    # Derivative of Ek
    Ek_dot = n / (1 - e * np.cos(Ek))
    vk_dot = np.sqrt(1 - e**2) * Ek_dot / (1 - e * np.cos(Ek))
    uk_dot = vk_dot + 2*(Cus*np.cos(2*pk) - Cuc*np.sin(2*pk))*vk_dot
    rk_dot = a*e*np.sin(Ek)*Ek_dot + 2*(Crs*np.cos(2*pk) - Crc*np.sin(2*pk))*vk_dot
    ik_dot = idot + 2*(Cis*np.cos(2*pk) - Cic*np.sin(2*pk))*vk_dot
    Ok_dot = odot - OMEGA_E

    xp_dot = rk_dot*np.cos(uk) - rk*np.sin(uk)*uk_dot
    yp_dot = rk_dot*np.sin(uk) + rk*np.cos(uk)*uk_dot

    v_x = xp_dot*np.cos(Ok) - yp_dot*np.cos(ik)*np.sin(Ok) + yp_dot*np.sin(ik)*ik_dot*np.sin(Ok) - Xk*Ok_dot
    v_y = xp_dot*np.sin(Ok) + yp_dot*np.cos(ik)*np.cos(Ok) - yp_dot*np.sin(ik)*ik_dot*np.cos(Ok) + Yk*Ok_dot
    v_z = yp_dot*np.sin(ik) + rk*np.sin(uk)*np.cos(ik)*ik_dot
    
    return pos_final, np.array([v_x, v_y, v_z]), sat_clk_bias

def solve_pv(sat_pos, sat_vel, pr, doppler, initial_guess, C=299792458.0, L1=1575.42e6):
    # Solve Position (WLS)
    x = np.array(initial_guess, dtype=float)
    for _ in range(10):
        H, dP, W = [], [], []
        for i in range(len(pr)):
            vec = x[:3] - sat_pos[i]; d = np.linalg.norm(vec)
            el = np.arcsin(abs(vec[2])/d)
            H.append([vec[0]/d, vec[1]/d, vec[2]/d, 1.0])
            dP.append(pr[i] - (d + x[3] + 2.3/np.sin(max(el,0.1))))
            W.append(np.sin(max(el,0.1))**2)
        H, dP, W = np.array(H), np.array(dP), np.diag(W)
        try:
            update = np.linalg.solve(H.T @ W @ H, H.T @ W @ dP)
            x += update
            if np.linalg.norm(update[:3]) < 1e-3: break
        except: return None, None

    # Solve Velocity
    v = np.zeros(4)
    H_v, dD = [], []
    for i in range(len(doppler)):
        vec = x[:3] - sat_pos[i]; d = np.linalg.norm(vec)
        unit = vec / d
        # Range Rate Observed (from Doppler)
        range_rate_obs = -doppler[i] * (C / L1)
        # Expected Range Rate (Satellite motion)
        range_rate_exp = np.dot(unit, sat_vel[i])
        dD.append(range_rate_obs - range_rate_exp)
        H_v.append([unit[0], unit[1], unit[2], 1.0])
    try:
        v = np.linalg.solve(np.array(H_v).T @ np.array(H_v), np.array(H_v).T @ np.array(dD))
    except: v = [0,0,0,0]
        
    return x, v

def main():
    C, LEAP, DIFF = 299792458.0, 18, 315964800 
    ISRAEL_CENTER = np.array([4438000.0, 3085000.0, 3369000.0, 0.0])
    
    print("Loading data...")
    obs_data = gr.load(r'rinex_files\gnss_log_2026_03_21_17_14_34.26o', use='G') 
    nav_data = gr.load(r'rinex_files\BRDC00IGS_R_20260800000_01D_MN.rnx', use='G')
    transformer = Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84'}, {"proj": 'latlong', "ellps": 'WGS84'})
    
    MAP = {
        'af0': ['af0', 'SVclockBias'], 'af1': ['af1', 'SVclockDrift'], 'af2': ['af2', 'SVclockDriftRate'],
        'Toe': ['Toe', 'time'], 'sqrtA': ['sqrtA'], 'Eccentricity': ['Eccentricity'], 'Io': ['Io'],
        'Omega0': ['Omega0'], 'omega': ['omega'], 'M0': ['M0'], 'DeltaN': ['DeltaN'],
        'OmegaDot': ['OmegaDot'], 'IDOT': ['IDOT'], 'Cuc': ['Cuc'], 'Cus': ['Cus'],
        'Crc': ['Crc'], 'Crs': ['Crs'], 'Cic': ['Cic'], 'Cis': ['Cis']
    }

    current_guess = ISRAEL_CENTER.copy()
    results = []

    print("Solving P, V, T...")
    for epoch_time in obs_data.time.values:
        obs_epoch = obs_data.sel(time=epoch_time)
        sat_p_list, sat_v_list, pr_list, dop_list = [], [], [], []
        t_utc = pd.to_datetime(epoch_time)
        tow = (t_utc.timestamp() - DIFF + LEAP) % 604800 

        for sv in obs_epoch.sv.values:
            pr = obs_epoch['C1C'].sel(sv=sv).values if 'C1C' in obs_epoch else obs_epoch['C1'].sel(sv=sv).values
            dop = obs_epoch['D1C'].sel(sv=sv).values if 'D1C' in obs_epoch else np.nan
            if not np.isnan(pr) and not np.isnan(dop) and sv in nav_data.sv:
                try:
                    nav_msg = nav_data.sel(sv=sv).dropna(dim='time', how='all').sel(time=epoch_time, method='nearest', tolerance=pd.Timedelta(hours=4))
                    n_d = {'pr': pr}
                    for k, opts in MAP.items():
                        for o in opts:
                            if o in nav_msg: n_d[k] = float(nav_msg[o].values); break
                    
                    s_p, s_v, s_c = calculate_satellite_data(n_d, tow - (pr/C))
                    sat_p_list.append(s_p); sat_v_list.append(s_v)
                    pr_list.append(pr + (s_c * C)); dop_list.append(dop)
                except: continue
                    
        if len(pr_list) >= 4:
            p, v = solve_pv(sat_p_list, sat_v_list, pr_list, dop_list, current_guess)
            if p is not None:
                current_guess = p
                lon, lat, alt = transformer.transform(p[0], p[1], p[2])
                results.append({'UTC': t_utc.strftime('%H:%M:%S'), 'Lat': lat, 'Lon': lon, 'Alt': alt, 'Vx': v[0], 'Vy': v[1], 'Vz': v[2]})

    if results:
        df = pd.DataFrame(results)
        df['Lat'] = df['Lat'].rolling(5, min_periods=1, center=True).mean()
        df['Lon'] = df['Lon'].rolling(5, min_periods=1, center=True).mean()
        df.to_csv("final_path_results.csv", index=False)
        
        kml = simplekml.Kml()
        kml.newlinestring(name="Final Path", coords=[(r['Lon'], r['Lat'], r['Alt']) for i, r in df.iterrows()])
        kml.save("final_path_results.kml")
        print(f"DONE: Exported {len(df)} epochs with Position and Velocity.")

if __name__ == "__main__":
    main()