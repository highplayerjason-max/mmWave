import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ObjectCluster:
    object_id: int
    x: float
    y: float
    z: float
    range_m: float
    azimuth_deg: float
    point_count: int
    peak_snr: Optional[int]
    mean_velocity: float


def _point_range(point) -> float:
    return math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)


def cluster_points(
    points,
    eps_m: float = 0.55,
    z_weight: float = 0.6,
    min_points: int = 2,
    max_objects: int = 6,
    y_min: float = 0.05,
    y_max: float = 6.0,
):
    """Group recent mmWave detected points into coarse object candidates.

    TI OOB detected points are sparse CFAR peaks, so this is intentionally a
    coarse human-readable grouping layer, not a semantic object detector.
    """
    usable = [
        point
        for point in points
        if y_min <= point.y <= y_max and abs(point.x) <= 3.0 and -2.0 <= point.z <= 2.0
    ]
    if not usable:
        return []

    n = len(usable)
    visited = [False] * n
    assigned = [False] * n
    clusters = []

    def dist(i: int, j: int) -> float:
        a = usable[i]
        b = usable[j]
        dx = a.x - b.x
        dy = a.y - b.y
        dz = (a.z - b.z) * z_weight
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    neighbors = [[j for j in range(n) if dist(i, j) <= eps_m] for i in range(n)]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_points:
            continue

        cluster_idx = []
        queue = list(neighbors[i])
        assigned[i] = True
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_points:
                    for k in neighbors[j]:
                        if k not in queue:
                            queue.append(k)
            if not assigned[j]:
                assigned[j] = True
                cluster_idx.append(j)

        if i not in cluster_idx:
            cluster_idx.append(i)
        clusters.append([usable[idx] for idx in cluster_idx])

    object_clusters = []
    for cluster in clusters:
        weights = [max(point.snr or 1, 1) for point in cluster]
        total_weight = sum(weights)
        x = sum(point.x * weight for point, weight in zip(cluster, weights)) / total_weight
        y = sum(point.y * weight for point, weight in zip(cluster, weights)) / total_weight
        z = sum(point.z * weight for point, weight in zip(cluster, weights)) / total_weight
        range_m = math.sqrt(x * x + y * y + z * z)
        peak_snr = max((point.snr for point in cluster if point.snr is not None), default=None)
        mean_velocity = sum(point.velocity for point in cluster) / len(cluster)
        object_clusters.append(
            ObjectCluster(
                object_id=0,
                x=x,
                y=y,
                z=z,
                range_m=range_m,
                azimuth_deg=math.degrees(math.atan2(x, y)),
                point_count=len(cluster),
                peak_snr=peak_snr,
                mean_velocity=mean_velocity,
            )
        )

    object_clusters.sort(key=lambda obj: (obj.peak_snr or 0, obj.point_count), reverse=True)
    for object_id, obj in enumerate(object_clusters[:max_objects], start=1):
        obj.object_id = object_id
    return object_clusters[:max_objects]
