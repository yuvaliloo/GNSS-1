# 🚀 GNSS Road Navigator

This project provides a robust, physics-based pipeline to process raw GNSS observation files (`.26o`) into clean, map-matched KML paths. It is specifically designed to handle the noise, multipath interference, and atmospheric delays typical of smartphone GNSS loggers in urban and highway environments.

## 📋 1. Prerequisites
This project requires Python 3.x. Install the necessary dependencies:

```bash
pip install georinex numpy pandas simplekml pyproj
 ```

## 📂 2. Project Structure

* **`gnss_log_XXXX.26o`**: Your raw Observation file from the Android GNSS Logger.
* **`BRDC...rnx`**: The Navigation file (Ephemeris data) containing satellite orbit information for the same timeframe.
* **`GNSS_road_navigator.py`**: The main processing script used to execute the pipeline.

## ⚙️ 3. Step-by-Step Logic
The pipeline operates in 5 distinct phases to transform raw radio measurements into a precise highway path:

### 1. Data Ingestion & Synchronization
* **Parsing:** The script parses the RINEX `.26o` (Observation) files and the BRDC Navigation file.
* **Synchronization:** It converts epoch timestamps into GPS Time of Week (TOW) to synchronize the receivers local clock with the satellite constellation time.

### ⚙️ 2. Advanced Physics Engine
We calculate the satellite positions using a high-fidelity model:

* **Keplerian Orbit Math**: Deriving precise satellite locations.
* **Sagnac Effect**: Applying a rotation matrix to compensate for Earth's rotation during the signal's flight time.
* **Relativistic Correction**: Applying the General Relativity clock drift correction (critical for meter-level accuracy).
* **Tropospheric Delay**: Using a zenith delay model combined with a mapping function to account for atmospheric signal slowing.

### ⚙️ 3. Position Solving & Integrity (RAIM)

* **Weighted Least Squares (WLS)**: The solver intersects satellite range spheres to mathematically determine your precise coordinates.
* **RAIM (Receiver Autonomous Integrity Monitoring)**: We enforce a 5-satellite minimum constraint. If the geometry yields high residuals, the solver identifies and drops the "liar" satellite (the one causing the most error) and re-solves, ensuring that a single bouncing signal cannot ruin the position.

### ⚙️ 4. Noise Filtering

* **Macro-Trend Filter**: A 31-second rolling median is used to find the "center of gravity" of your path. Any raw point deviating > 100m from this macro-trend is ruthlessly deleted to kill ocean/building spikes.
* **Micro-Median Filter**: A 7-second rolling median cleans up local jitters.
* **Smoothing**: A 3-second moving average curves the path naturally to match road geometry.

### ⚙️ 5. Vector Translation (The "Shadow" Snap)
Atmospheric conditions often create a systematic bias ("shadow") where the track is perfect in shape but shifted from the road.

* **Static Vector Shift**: The script applies a calculated coordinate translation to snap the entire dataset onto the correct pavement.

## 🛠️ 4. Troubleshooting the "Shadow"
If your path is perfectly shaped but shifted:

1. Open your generated `gnss_full_road.kml` in Google Earth.
2. Use the **Ruler Tool** to measure the distance in meters between your path and the actual road.
3. Update the `SHIFT_NORTH_METERS` and `SHIFT_EAST_METERS` variables in the Post-Processing section of the script.
4. Re-run the script.

## 🚀 5. Usage

1. Place your data files in the `rinex_files` directory.
2. Ensure the file paths in the `main()` function point to your files.
3. Execute the script:

```bash
python GNSS_road_navigator.py
```