import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from typing import List, Dict, Any, Tuple

class StarImageProcessingPipeline:
    """
    Image processing pipeline for star identification from raw sensor images.
    Performs denoising, thresholding, centroid extraction, and brightness estimation.
    """
    def __init__(self, sigma: float = 1.0, threshold: float = 0.5):
        self.sigma = sigma
        self.threshold = threshold

    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # 1. Denoise (Gaussian blur)
        filtered = gaussian_filter(image, self.sigma)
        # 2. Threshold to find star candidates
        binary = filtered > self.threshold * np.max(filtered)
        # 3. Label connected components
        labeled, num_features = label(binary)
        # 4. Extract star centroids and brightness
        stars = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            if np.sum(mask) == 0:
                continue
            centroid = center_of_mass(filtered, labels=labeled, index=i)
            brightness = np.sum(filtered[mask])
            stars.append({
                'centroid': np.array(centroid),
                'brightness': brightness,
                'area': np.sum(mask)
            })
        return stars

    def extract_vectors(self, stars: List[Dict[str, Any]], focal_length_px: float, image_shape: Tuple[int, int]) -> List[np.ndarray]:
        # Convert centroids to unit vectors in camera frame
        cx, cy = image_shape[1] / 2, image_shape[0] / 2
        vectors = []
        for s in stars:
            x, y = s['centroid'][1] - cx, s['centroid'][0] - cy
            z = focal_length_px
            v = np.array([x, y, z])
            v /= np.linalg.norm(v)
            vectors.append(v)
        return vectors
