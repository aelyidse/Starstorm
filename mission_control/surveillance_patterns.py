import numpy as np
from typing import List, Tuple, Dict, Any

class AreaSurveillancePatterns:
    """
    Generates waypoint sequences for area surveillance using standard search patterns.
    Supports lawnmower, expanding square, spiral, and custom polygonal sweeps.
    """
    def __init__(self):
        pass

    def lawnmower(self, area_bounds: Tuple[float, float, float, float], track_spacing: float) -> List[Tuple[float, float]]:
        # area_bounds: (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = area_bounds
        waypoints = []
        y_tracks = np.arange(ymin, ymax, track_spacing)
        for i, y in enumerate(y_tracks):
            if i % 2 == 0:
                waypoints.append((xmin, y))
                waypoints.append((xmax, y))
            else:
                waypoints.append((xmax, y))
                waypoints.append((xmin, y))
        return waypoints

    def expanding_square(self, center: Tuple[float, float], spacing: float, n_legs: int) -> List[Tuple[float, float]]:
        # Generates waypoints in an expanding square pattern
        x, y = center
        waypoints = [(x, y)]
        for i in range(1, n_legs + 1):
            waypoints.append((x + i * spacing, y))
            waypoints.append((x + i * spacing, y + i * spacing))
            waypoints.append((x - i * spacing, y + i * spacing))
            waypoints.append((x - i * spacing, y - i * spacing))
            waypoints.append((x + i * spacing, y - i * spacing))
        return waypoints

    def spiral(self, center: Tuple[float, float], max_radius: float, n_points: int) -> List[Tuple[float, float]]:
        # Generates a spiral pattern (Archimedean)
        x0, y0 = center
        waypoints = []
        for i in range(n_points):
            theta = 2 * np.pi * i / n_points * (max_radius / 10)
            r = max_radius * i / n_points
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            waypoints.append((x, y))
        return waypoints

    def polygonal_sweep(self, vertices: List[Tuple[float, float]], spacing: float) -> List[Tuple[float, float]]:
        # Generates waypoints for a polygonal area sweep (simple raster fill)
        # For simplicity, only works for convex polygons
        from shapely.geometry import Polygon, LineString
        poly = Polygon(vertices)
        minx, miny, maxx, maxy = poly.bounds
        waypoints = []
        y = miny
        while y <= maxy:
            line = LineString([(minx, y), (maxx, y)])
            inter = poly.intersection(line)
            if inter.is_empty:
                y += spacing
                continue
            if inter.geom_type == 'MultiLineString':
                for seg in inter:
                    waypoints.append(seg.coords[0])
                    waypoints.append(seg.coords[-1])
            elif inter.geom_type == 'LineString':
                waypoints.append(inter.coords[0])
                waypoints.append(inter.coords[-1])
            y += spacing
        return waypoints
