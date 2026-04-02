import pynmea2
import pandas as pd
import simplekml

def robust_nmea_parse(filename):
    results = []
    print(f"--- Processing GNSSLogger NMEA format ---")
    
    with open(filename, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # 1. Handle the "NMEA," prefix by finding the first '$'
            if '$' in line:
                try:
                    # Slices from the first '$' to the end of the line
                    clean_line = line[line.find('$'):]
                    
                    # 2. Parse the sentence
                    msg = pynmea2.parse(clean_line)
                    
                    # 3. Check if this specific NMEA sentence contains coordinates
                    # (GGA, RMC, and GNS messages have these attributes)
                    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                        # Only keep points where coordinates are not zero
                        if msg.latitude != 0 and msg.longitude != 0:
                            
                            # 4. Safely handle the timestamp 
                            # (Some sentences log coordinates but skip the time field)
                            if hasattr(msg, 'timestamp') and msg.timestamp:
                                time_str = msg.timestamp.strftime('%H:%M:%S')
                            else:
                                time_str = "N/A"
                                
                            # 5. Safely handle Altitude (Only found in GGA/GNS)
                            alt = getattr(msg, 'altitude', 0.0)
                            
                            results.append({
                                'Time': time_str,
                                'Lat': msg.latitude,
                                'Lon': msg.longitude,
                                'Alt': alt
                            })
                except Exception:
                    # Skip lines that are malformed or have bad checksums
                    continue

    if results:
        # Convert to DataFrame and save CSV
        df = pd.DataFrame(results)
        df.to_csv("ground_truth_final.csv", index=False)
        
        # 6. Export KML for Google Earth
        kml = simplekml.Kml()
        # KML uses (Longitude, Latitude, Altitude) order
        coords = [(row['Lon'], row['Lat'], row['Alt']) for _, row in df.iterrows()]
        
        line_feat = kml.newlinestring(name="NMEA Ground Truth")
        line_feat.coords = coords
        line_feat.style.linestyle.color = 'ff00ff00' # Green (AABBGGRR)
        line_feat.style.linestyle.width = 5
        
        kml.save("ground_truth_final.kml")
        
        print(f"--- SUCCESS: Extracted {len(df)} points! ---")
        print("Files saved: ground_truth_final.csv, ground_truth_final.kml")
    else:
        print("--- FAILURE: Still could not find coordinates. ---")
        print("Double check that your file contains lines like 'NMEA,$GNGGA...' or 'NMEA,$GNRMC...'")


robust_nmea_parse('gnss_log_2026_03_21_17_14_34.nmea')