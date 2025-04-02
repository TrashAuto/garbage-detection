import math
from rplidar import RPLidar
from time import sleep

def polar_to_cartesian(angle_deg, distance_mm):
    angle_rad = math.radians(angle_deg)
    x = distance_mm * math.cos(angle_rad)
    y = distance_mm * math.sin(angle_rad)
    return (x, y)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_object_of_interest(
    min_physical_width_mm=25,
    big_object_threshold_mm=150,
    max_gap_mm=300,
    max_distance=2000,
    min_points=2,
    min_proximity_mm=500,
    max_attempts=4,
    merge_cluster_threshold_mm=150
):
    PORT_NAME = '/dev/ttyUSB0'
    lidar = RPLidar(PORT_NAME)
    lidar.start_motor()
    sleep(1)

    attempt = 0

    print(f"\n Scanning for objects in the 225 deg to 315 deg field of view... (max {max_attempts} attempts)")

    try:
        scan_points = []

        for scan in lidar.iter_measures():
            quality, _, angle, distance = scan
            if 0 < distance <= max_distance:
                scan_points.append((angle % 360, distance))

            if angle % 360 < 1.0:
                attempt += 1

                filtered_points = [pt for pt in scan_points if 225 <= pt[0] <= 315]
                filtered_points.sort()

                cartesian_points = [(ang, polar_to_cartesian(ang, dist), dist) for ang, dist in filtered_points]

                # Initial distance-based clustering
                clusters = []
                current_cluster = []

                for i in range(len(cartesian_points)):
                    if not current_cluster:
                        current_cluster.append(cartesian_points[i])
                    else:
                        dist = euclidean_distance(current_cluster[-1][1], cartesian_points[i][1])
                        if dist <= max_gap_mm:
                            current_cluster.append(cartesian_points[i])
                        else:
                            if len(current_cluster) >= min_points:
                                clusters.append(current_cluster)
                            current_cluster = [cartesian_points[i]]

                if current_cluster and len(current_cluster) >= min_points:
                    clusters.append(current_cluster)

                # Merge nearby clusters
                merged = [False] * len(clusters)
                final_clusters = []

                for i in range(len(clusters)):
                    if merged[i]:
                        continue
                    merged[i] = True
                    base_cluster = clusters[i]
                    base_points = [pt[1] for pt in base_cluster]
                    base_center = (
                        sum([p[0] for p in base_points]) / len(base_points),
                        sum([p[1] for p in base_points]) / len(base_points)
                    )
                    for j in range(i + 1, len(clusters)):
                        if merged[j]:
                            continue
                        compare_points = [pt[1] for pt in clusters[j]]
                        compare_center = (
                            sum([p[0] for p in compare_points]) / len(compare_points),
                            sum([p[1] for p in compare_points]) / len(compare_points)
                        )
                        if euclidean_distance(base_center, compare_center) <= merge_cluster_threshold_mm:
                            base_cluster += clusters[j]
                            merged[j] = True
                    final_clusters.append(base_cluster)

                # Analyze merged clusters
                for cluster in final_clusters:
                    angles = [pt[0] for pt in cluster]
                    distances = [pt[2] for pt in cluster]

                    angles_sorted = sorted(angles)

                    if len(angles_sorted) >= 2:
                        diffs = [angles_sorted[i+1] - angles_sorted[i] for i in range(len(angles_sorted)-1)]
                        avg_step = sum(diffs) / len(diffs)
                        estimated_from_steps = avg_step * (len(angles_sorted) - 1)
                        direct_span = angles_sorted[-1] - angles_sorted[0]
                        estimated_span = max(estimated_from_steps, direct_span)
                    else:
                        estimated_span = 0

                    avg_distance = sum(distances) / len(distances)
                    if avg_distance <= 250 and estimated_span < 20:
                        estimated_span = 20  # enforce a minimum span for close objects

                    center_angle = (angles_sorted[0] + estimated_span / 2) % 360
                    closest = min(distances)

                    width = 2 * avg_distance * math.tan(math.radians(estimated_span / 2))

                    # print(f"   span: {estimated_span:.1f} deg, width: {width:.1f} mm, points: {len(cluster)}")

                    if width < min_physical_width_mm:
                        continue

                    size_class = "big" if width >= big_object_threshold_mm else "small"

                    relative_angle = (center_angle - 270) % 360
                    if relative_angle > 180:
                        direction = f"{360 - int(relative_angle)} degrees right from zero"
                    else:
                        direction = f"{int(relative_angle)} degrees left from zero"

                    print(f" -> Detected object: ({size_class}, {int(avg_distance)} mm, {direction})")

                    if size_class == "small" and avg_distance <= min_proximity_mm:
                        return {
                            'width_mm': int(width),
                            'distance_mm': int(avg_distance),
                            'angle_center_deg': round(center_angle, 1),
                            'size_class': size_class
                        }

                if attempt >= max_attempts:
                    print("Max attempts reached. No valid object detected.")
                    break

                scan_points = []

    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    return None
