import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, List, Optional
import math

class ReferenceFrame(Enum):
    """Reference frames used in space simulations"""
    ECI = "Earth-Centered Inertial"  # Fixed relative to stars
    ECEF = "Earth-Centered Earth-Fixed"  # Fixed relative to Earth's surface
    LVLH = "Local Vertical Local Horizontal"  # Orbital reference frame
    BODY = "Body-Fixed"  # Fixed to spacecraft body
    PERIFOCAL = "Perifocal"  # Orbital plane reference frame
    GEODETIC = "Geodetic"  # Latitude, longitude, altitude
    SPHERICAL = "Spherical"  # r, θ, φ coordinates

class CoordinateSystem:
    """
    Unified coordinate system with reference frame transformations.
    Provides consistent coordinate representations and transformations
    between different reference frames used in space simulations.
    """
    def __init__(self, earth_radius_m: float = 6371000.0, earth_mu: float = 3.986004418e14):
        self.earth_radius_m = earth_radius_m  # Earth radius in meters
        self.earth_mu = earth_mu  # Earth's gravitational parameter (m³/s²)
        
    def eci_to_ecef(self, position: np.ndarray, velocity: np.ndarray, 
                   time_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from Earth-Centered Inertial to Earth-Centered Earth-Fixed frame.
        
        Args:
            position: Position vector [x, y, z] in ECI frame (m)
            velocity: Velocity vector [vx, vy, vz] in ECI frame (m/s)
            time_s: Time since epoch (s)
            
        Returns:
            Tuple of (position, velocity) in ECEF frame
        """
        # Earth's rotation rate (rad/s)
        omega_earth = 7.2921150e-5
        
        # Rotation angle
        theta = omega_earth * time_s
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Angular velocity vector
        omega_vec = np.array([0, 0, omega_earth])
        
        # Transform position
        pos_ecef = R @ position
        
        # Transform velocity (includes Coriolis effect)
        vel_ecef = R @ velocity - np.cross(omega_vec, pos_ecef)
        
        return pos_ecef, vel_ecef
    
    def ecef_to_eci(self, position: np.ndarray, velocity: np.ndarray, 
                   time_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from Earth-Centered Earth-Fixed to Earth-Centered Inertial frame.
        
        Args:
            position: Position vector [x, y, z] in ECEF frame (m)
            velocity: Velocity vector [vx, vy, vz] in ECEF frame (m/s)
            time_s: Time since epoch (s)
            
        Returns:
            Tuple of (position, velocity) in ECI frame
        """
        # Earth's rotation rate (rad/s)
        omega_earth = 7.2921150e-5
        
        # Rotation angle
        theta = omega_earth * time_s
        
        # Rotation matrix (transpose of ECEF to ECI)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Angular velocity vector
        omega_vec = np.array([0, 0, omega_earth])
        
        # Transform position
        pos_eci = R @ position
        
        # Transform velocity (includes Coriolis effect)
        vel_eci = R @ velocity + np.cross(omega_vec, position)
        
        return pos_eci, vel_eci
    
    def eci_to_perifocal(self, position: np.ndarray, velocity: np.ndarray,
                        i: float, omega: float, w: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from ECI to perifocal frame (orbital plane coordinates).
        
        Args:
            position: Position vector [x, y, z] in ECI frame (m)
            velocity: Velocity vector [vx, vy, vz] in ECI frame (m/s)
            i: Inclination (rad)
            omega: Right ascension of ascending node (rad)
            w: Argument of periapsis (rad)
            
        Returns:
            Tuple of (position, velocity) in perifocal frame
        """
        # Rotation matrix from ECI to perifocal
        R11 = np.cos(omega) * np.cos(w) - np.sin(omega) * np.sin(w) * np.cos(i)
        R12 = np.sin(omega) * np.cos(w) + np.cos(omega) * np.sin(w) * np.cos(i)
        R13 = np.sin(w) * np.sin(i)
        
        R21 = -np.cos(omega) * np.sin(w) - np.sin(omega) * np.cos(w) * np.cos(i)
        R22 = -np.sin(omega) * np.sin(w) + np.cos(omega) * np.cos(w) * np.cos(i)
        R23 = np.cos(w) * np.sin(i)
        
        R31 = np.sin(omega) * np.sin(i)
        R32 = -np.cos(omega) * np.sin(i)
        R33 = np.cos(i)
        
        R = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
        
        # Transform position and velocity
        pos_perifocal = R @ position
        vel_perifocal = R @ velocity
        
        return pos_perifocal, vel_perifocal
    
    def eci_to_lvlh(self, position: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform from ECI to Local Vertical Local Horizontal frame.
        
        Args:
            position: Position vector [x, y, z] in ECI frame (m)
            velocity: Velocity vector [vx, vy, vz] in ECI frame (m/s)
            
        Returns:
            Tuple of (position, velocity) in LVLH frame
        """
        # Normalize position vector (radial direction)
        r_norm = np.linalg.norm(position)
        r_unit = position / r_norm
        
        # Angular momentum vector
        h_vec = np.cross(position, velocity)
        h_unit = h_vec / np.linalg.norm(h_vec)
        
        # Complete the right-handed system
        theta_unit = np.cross(h_unit, r_unit)
        
        # Rotation matrix from ECI to LVLH
        R = np.vstack((r_unit, theta_unit, h_unit))
        
        # Transform position and velocity
        pos_lvlh = np.array([r_norm, 0, 0])  # In LVLH, position is always along radial axis
        vel_lvlh = R @ velocity
        
        return pos_lvlh, vel_lvlh
    
    def geodetic_to_ecef(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF Cartesian coordinates.
        
        Args:
            lat_deg: Latitude in degrees
            lon_deg: Longitude in degrees
            alt_m: Altitude above reference ellipsoid in meters
            
        Returns:
            Position vector [x, y, z] in ECEF frame (m)
        """
        # Convert to radians
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis (m)
        f = 1/298.257223563  # flattening
        e2 = 2*f - f*f  # square of eccentricity
        
        # Normal radius of curvature
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # ECEF coordinates
        x = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt_m) * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def ecef_to_geodetic(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert ECEF Cartesian coordinates to geodetic coordinates.
        
        Args:
            position: Position vector [x, y, z] in ECEF frame (m)
            
        Returns:
            Tuple of (latitude_deg, longitude_deg, altitude_m)
        """
        x, y, z = position
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # semi-major axis (m)
        f = 1/298.257223563  # flattening
        b = a * (1 - f)  # semi-minor axis
        e2 = 2*f - f*f  # square of eccentricity
        
        # Longitude is straightforward
        lon_rad = np.arctan2(y, x)
        
        # Distance from Z-axis
        p = np.sqrt(x**2 + y**2)
        
        # Initial latitude guess
        lat_rad = np.arctan2(z, p * (1 - e2))
        
        # Iterative solution for latitude
        for _ in range(5):  # Usually converges in 2-3 iterations
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
            h = p / np.cos(lat_rad) - N
            lat_rad = np.arctan2(z, p * (1 - e2 * N / (N + h)))
        
        # Final altitude calculation
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        h = p / np.cos(lat_rad) - N
        
        return np.degrees(lat_rad), np.degrees(lon_rad), h
    
    def body_to_eci(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Transform from body-fixed frame to ECI frame using quaternion rotation.
        
        Args:
            position: Position vector [x, y, z] in body frame (m)
            quaternion: Orientation quaternion [w, x, y, z]
            
        Returns:
            Position vector in ECI frame
        """
        # Extract quaternion components
        q0, q1, q2, q3 = quaternion
        
        # Construct rotation matrix from quaternion
        R = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])
        
        # Transform position
        return R @ position
    
    def eci_to_body(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Transform from ECI frame to body-fixed frame using quaternion rotation.
        
        Args:
            position: Position vector [x, y, z] in ECI frame (m)
            quaternion: Orientation quaternion [w, x, y, z]
            
        Returns:
            Position vector in body frame
        """
        # Extract quaternion components
        q0, q1, q2, q3 = quaternion
        
        # Construct rotation matrix from quaternion (transpose of body to ECI)
        R = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3+q0*q1)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), 1-2*(q1**2+q2**2)]
        ])
        
        # Transform position
        return R @ position
    
    def orbital_elements_to_eci(self, a: float, e: float, i: float, 
                              omega: float, w: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to ECI position and velocity.
        
        Args:
            a: Semi-major axis (m)
            e: Eccentricity
            i: Inclination (rad)
            omega: Right ascension of ascending node (rad)
            w: Argument of periapsis (rad)
            nu: True anomaly (rad)
            
        Returns:
            Tuple of (position, velocity) in ECI frame
        """
        # Calculate position and velocity in perifocal frame
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        
        pos_perifocal = np.array([
            r * np.cos(nu),
            r * np.sin(nu),
            0
        ])
        
        vel_perifocal = np.sqrt(self.earth_mu / p) * np.array([
            -np.sin(nu),
            e + np.cos(nu),
            0
        ])
        
        # Rotation matrix from perifocal to ECI
        R11 = np.cos(omega) * np.cos(w) - np.sin(omega) * np.sin(w) * np.cos(i)
        R12 = -np.cos(omega) * np.sin(w) - np.sin(omega) * np.cos(w) * np.cos(i)
        R13 = np.sin(omega) * np.sin(i)
        
        R21 = np.sin(omega) * np.cos(w) + np.cos(omega) * np.sin(w) * np.cos(i)
        R22 = -np.sin(omega) * np.sin(w) + np.cos(omega) * np.cos(w) * np.cos(i)
        R23 = -np.cos(omega) * np.sin(i)
        
        R31 = np.sin(w) * np.sin(i)
        R32 = np.cos(w) * np.sin(i)
        R33 = np.cos(i)
        
        R = np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
        
        # Transform to ECI
        pos_eci = R @ pos_perifocal
        vel_eci = R @ vel_perifocal
        
        return pos_eci, vel_eci
    
    def eci_to_orbital_elements(self, position: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
        """
        Convert ECI position and velocity to orbital elements.
        
        Args:
            position: Position vector [x, y, z] in ECI frame (m)
            velocity: Velocity vector [vx, vy, vz] in ECI frame (m/s)
            
        Returns:
            Dictionary of orbital elements
        """
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        # Angular momentum vector
        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)
        
        # Node vector
        n_vec = np.cross(np.array([0, 0, 1]), h_vec)
        n = np.linalg.norm(n_vec)
        
        # Eccentricity vector
        e_vec = ((v**2 - self.earth_mu/r) * position - np.dot(position, velocity) * velocity) / self.earth_mu
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        a = h**2 / (self.earth_mu * (1 - e**2)) if e < 1.0 else -self.earth_mu / (v**2 - 2*self.earth_mu/r)
        
        # Inclination
        i = np.arccos(h_vec[2] / h)
        
        # Right ascension of ascending node
        omega = np.arccos(n_vec[0] / n) if n > 0 else 0.0
        if n_vec[1] < 0:
            omega = 2 * np.pi - omega
            
        # Argument of periapsis
        w = np.arccos(np.dot(n_vec, e_vec) / (n * e)) if n > 0 and e > 0 else 0.0
        if e_vec[2] < 0:
            w = 2 * np.pi - w
            
        # True anomaly
        nu = np.arccos(np.dot(e_vec, position) / (e * r)) if e > 0 else 0.0
        if np.dot(position, velocity) < 0:
            nu = 2 * np.pi - nu
            
        return {
            'semi_major_axis_m': a,
            'eccentricity': e,
            'inclination_rad': i,
            'raan_rad': omega,
            'arg_periapsis_rad': w,
            'true_anomaly_rad': nu,
            'period_s': 2 * np.pi * np.sqrt(a**3 / self.earth_mu) if a > 0 else float('nan')
        }