from gpiozero import DistanceSensor
import warnings
warnings.filterwarnings("ignore")

sensor = DistanceSensor(echo=24, trigger=23, max_distance=2.5)

def is_tall_object_present(lidar_distance_mm, tolerance_mm=150):
    """
    Returns True if the ultrasonic sensor detects an object near the LiDAR's distance.
    """
    measured_mm = sensor.distance * 1000
    print(f"[Ultrasonic] Sensor measured distance: {measured_mm:.0f} mm")

    if abs(measured_mm - lidar_distance_mm) <= tolerance_mm:
        return True
    return False

