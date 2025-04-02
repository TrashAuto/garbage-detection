import time
from lidar_util import detect_object_of_interest
from ml_util import run_ml_pipeline
from sonic_util import is_tall_object_present  # Imported ultrasonic checker

# Configuration constants
DEBOUNCE_SECONDS = 10
SCAN_INTERVAL_SECONDS = 5
PROXIMITY_THRESHOLD_MM = 250
TALL_OBJECT_TOLERANCE_MM = 150  # How close ultrasonic reading must be to LiDAR distance

def main():
    print("Starting integrated LiDAR + Ultrasonic + ML detection loop...")
    last_trigger_time = 0

    while True:
        print("\n[LiDAR Scan Cycle]")
        result = detect_object_of_interest(
            min_physical_width_mm=20,
            big_object_threshold_mm=150,
            max_gap_mm=200,
            max_distance=2000,
            min_points=2,
            min_proximity_mm=PROXIMITY_THRESHOLD_MM,
            max_attempts=4
        )

        if result:
            current_time = time.time()
            print("Result passed from LiDAR:", result)

            if current_time - last_trigger_time >= DEBOUNCE_SECONDS:
                object_distance = result['distance_mm']

                print("Checking for tall object using ultrasonic sensor...")
                if is_tall_object_present(object_distance, TALL_OBJECT_TOLERANCE_MM):
                    print("Detected tall object! skipping ML to avoid false classification.")
                else:
                    print("No tall object detected! running ML classification...")
                    run_ml_pipeline()
                    last_trigger_time = current_time
            else:
                print("Debounce active! skipping ML trigger.")
        else:
            print("No valid small object detected.")

        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
