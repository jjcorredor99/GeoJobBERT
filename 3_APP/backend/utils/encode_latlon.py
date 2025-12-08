import json
import numpy as np

EARTH_R_KM = 6371.0088

def latlon_to_unit_xyz(lat_deg, lon_deg):
    """Vectorized: lat_deg, lon_deg can be scalars, 1D arrays, or Nx1 arrays."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)  # (..., 3)

class GeoFourierEncoder:
    """
    Sphere-aware Random Fourier Features for geo coords.
    D: output dimensionality (must be even); default D=8 -> m=4 frequency rows.
    scales_km: characteristic distance scales you care about (coarse→fine).
               Provide 4 numbers for D=8; ex: (5000, 2000, 700, 250).
    seed: makes frequencies reproducible; save/load self.B for exact reuse.
    """
    def __init__(self, D=8, scales_km=(5000, 2000, 700, 250), seed=0):
        assert D % 2 == 0, "D must be even (we emit sin+cos pairs)."
        self.D = D
        self.m = D // 2
        self.scales_km = np.array(scales_km, dtype=float).ravel()
        if self.scales_km.size == 1:
            self.scales_km = np.repeat(self.scales_km, self.m)
        elif self.scales_km.size < self.m:
            # Log-space interpolate to fill to length m
            t = np.linspace(0, 1, self.m)
            tin = np.linspace(0, 1, self.scales_km.size)
            self.scales_km = np.exp(np.interp(t, tin, np.log(self.scales_km)))
        elif self.scales_km.size > self.m:
            self.scales_km = self.scales_km[:self.m]
        self.seed = seed
        self.B = self._build_B()

    def _build_B(self):
        rng = np.random.default_rng(self.seed)
        # Convert desired km scales to chord-length on the unit sphere, then to Gaussian std
        theta = self.scales_km / EARTH_R_KM                    # central angle in radians
        ell_chord = 2.0 * np.sin(theta / 2.0)                  # chord length on unit sphere
        sigma = 1.0 / np.maximum(ell_chord, 1e-8)              # ω ~ N(0, 1/ell^2)
        B = rng.normal(loc=0.0, scale=sigma[:, None], size=(self.m, 3))
        return B.astype(np.float32)

    def transform(self, latlon_deg):
        """
        latlon_deg: array-like of shape (N, 2) in degrees (lat, lon).
        Returns: (N, D) features in [-1,1], L2-norm ~ O(1).
        """
        arr = np.asarray(latlon_deg, dtype=np.float32)
        xyz = latlon_to_unit_xyz(arr[:, 0], arr[:, 1])         # (N,3)
        proj = xyz @ self.B.T                                  # (N,m)
        feat = np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)
        feat *= (1.0 / np.sqrt(self.m))
        return feat

