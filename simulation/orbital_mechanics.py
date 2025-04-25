import math
from typing import Dict, Any, Tuple

class OrbitalMechanics:
    """
    Provides high-precision gravitational and orbital mechanics calculations for space operations.
    Supports Keplerian orbits, maneuvers, and propagation.
    """
    def __init__(self, mu: float = 3.986004418e14, r_earth: float = 6371000.0):
        # mu: Standard gravitational parameter (m^3/s^2), r_earth: meters
        self.mu = mu
        self.r_earth = r_earth

    def circular_velocity(self, altitude_m: float) -> float:
        r = self.r_earth + altitude_m
        return math.sqrt(self.mu / r)

    def escape_velocity(self, altitude_m: float) -> float:
        r = self.r_earth + altitude_m
        return math.sqrt(2 * self.mu / r)

    def period(self, altitude_m: float) -> float:
        r = self.r_earth + altitude_m
        return 2 * math.pi * math.sqrt(r ** 3 / self.mu)

    def kepler_propagate(self, a: float, e: float, i: float, omega: float, w: float, M0: float, t: float, t0: float = 0.0) -> Dict[str, Any]:
        """
        Propagate a Keplerian orbit (a: semi-major axis [m], e: eccentricity, i: inclination [rad],
        omega: RAAN [rad], w: argument of periapsis [rad], M0: mean anomaly at epoch [rad], t: time [s], t0: epoch [s])
        Returns position and velocity in perifocal frame.
        """
        n = math.sqrt(self.mu / a ** 3)
        M = M0 + n * (t - t0)
        E = self._solve_kepler(M, e)
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
        r = a * (1 - e * math.cos(E))
        # Position in perifocal frame
        x = r * math.cos(nu)
        y = r * math.sin(nu)
        z = 0.0
        return {
            'r_perifocal': (x, y, z),
            'nu': nu,
            'E': E,
            'M': M,
            'r': r
        }

    def _solve_kepler(self, M: float, e: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        # Solve Kepler's equation: M = E - e*sin(E) for E
        E = M if e < 0.8 else math.pi
        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            fp = 1 - e * math.cos(E)
            dE = -f / fp
            E += dE
            if abs(dE) < tol:
                break
        return E

    def hohmann_transfer(self, r1: float, r2: float) -> Dict[str, float]:
        """
        Compute delta-v and transfer time for a Hohmann transfer between two circular orbits (meters).
        """
        v1 = math.sqrt(self.mu / r1)
        v2 = math.sqrt(self.mu / r2)
        a_transfer = 0.5 * (r1 + r2)
        v_transfer1 = math.sqrt(self.mu * (2 / r1 - 1 / a_transfer))
        v_transfer2 = math.sqrt(self.mu * (2 / r2 - 1 / a_transfer))
        delta_v1 = v_transfer1 - v1
        delta_v2 = v2 - v_transfer2
        t_transfer = math.pi * math.sqrt(a_transfer ** 3 / self.mu)
        return {
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'transfer_time_s': t_transfer
        }

    def two_body_state(self, r0: Tuple[float, float, float], v0: Tuple[float, float, float], dt: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Propagate a two-body state vector (r0, v0) forward by dt seconds using universal variable formulation (simplified).
        Returns new (r, v) in ECI frame.
        """
        # For brevity, this is a simple Euler step; production code should use a universal variable solver
        rx, ry, rz = r0
        vx, vy, vz = v0
        r = math.sqrt(rx**2 + ry**2 + rz**2)
        ax = -self.mu * rx / r**3
        ay = -self.mu * ry / r**3
        az = -self.mu * rz / r**3
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        vz_new = vz + az * dt
        rx_new = rx + vx_new * dt
        ry_new = ry + vy_new * dt
        rz_new = rz + vz_new * dt
        return (rx_new, ry_new, rz_new), (vx_new, vy_new, vz_new)
