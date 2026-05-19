import pandas as pd
import numpy as np
import simplekml

def parse_nmea(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # The NMEA file you provided has a prefix, so we split by comma
            parts = line.strip().split(',')
            
            # $GNRMC contains the recommended minimum data (Lat, Lon, Speed)
            if len(parts) > 8 and (parts[1] == '$GNRMC' or parts[1] == '$GPRMC'):
                if parts[3] == 'A':  # 'A' means Active/Valid signal
                    
                    # NMEA format is ddmm.mmmmmm
                    lat_raw = float(parts[4])
                    lat_dir = parts[5]
                    lon_raw = float(parts[6])
                    lon_dir = parts[7]
                    
                    # Convert to Decimal Degrees
                    lat = int(lat_raw / 100) + (lat_raw % 100) / 60.0
                    if lat_dir == 'S': lat = -lat
                        
                    lon = int(lon_raw / 100) + (lon_raw % 100) / 60.0
                    if lon_dir == 'W': lon = -lon
                        
                    data.append({'Lat': lat, 'Lon': lon})
                    
    return pd.DataFrame(data)

def apply_kalman_loop(df):
    smoothed_lat = []
    smoothed_lon = []
    
    # 1. INITIALIZE: Start at the first point
    lat_est = df['Lat'].iloc[0]
    lon_est = df['Lon'].iloc[0]
    
    # The Error Covariance (How uncertain we are)
    P = 1.0 
    
    # Q = Process Noise (How much we trust the car's inertia / prediction)
    # R = Measurement Noise (How much we distrust the wandering GPS)
    # Tweak R higher if the line is too squiggly. Tweak R lower if it lags behind corners.
    Q = 1e-5
    R = 0.0001 
    
    # 2. THE GRADUAL FIX LOOP
    for i in range(len(df)):
        meas_lat = df['Lat'].iloc[i]
        meas_lon = df['Lon'].iloc[i]
        
        # PREDICTION STEP
        # (Assuming steady state for simplicity, error grows slightly)
        P = P + Q
        
        # UPDATE STEP (The "Gradual Fix")
        # Calculate the Kalman Gain: K = P / (P + R)
        K = P / (P + R)
        
        # Nudge the estimation toward the measurement based on the Gain
        lat_est = lat_est + K * (meas_lat - lat_est)
        lon_est = lon_est + K * (meas_lon - lon_est)
        
        # Update the error covariance
        P = (1 - K) * P
        
        smoothed_lat.append(lat_est)
        smoothed_lon.append(lon_est)
        
    df['Smooth_Lat'] = smoothed_lat
    df['Smooth_Lon'] = smoothed_lon
    return df

def main():
    # Make sure this points to your .nmea / .txt file!
    nmea_file = 'rinex_files\gnss_log_2026_03_22_08_44_20.nmea' 
    
    print("Parsing NMEA data...")
    df = parse_nmea(nmea_file)
    
    if df.empty:
        print("Error: No valid $GNRMC sentences found in file.")
        return
        
    print(f"Loaded {len(df)} raw points. Applying your gradual fixing loop...")
    
    # Apply the Kalman Filter
    df = apply_kalman_loop(df)
    
    # Export to KML
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="Kalman Smoothed Path")
    ls.coords = [(r['Smooth_Lon'], r['Smooth_Lat']) for i, r in df.iterrows()] 
    ls.tessellate = 1
    ls.style.linestyle.color = 'ff00ffff'  # Yellow/Magenta line
    ls.style.linestyle.width = 4          
    
    kml.save("nmea_smoothed_road.kml")
    print(f"SUCCESS: Exported {len(df)} fixed points to KML.")

if __name__ == "__main__":
    main()