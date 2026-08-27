import cv2
import numpy as np

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization."""
    if image.ndim > 2:
        # Equalize each band separately
        equalized = np.zeros_like(image)
        for i in range(image.shape[-1]):
            equalized[..., i] = histogram_equalization(image[..., i])
        return equalized
        
    if image.dtype != np.uint8:
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            img_uint8 = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.zeros_like(image, dtype=np.uint8)
    else:
        img_uint8 = image
        
    return cv2.equalizeHist(img_uint8)

def normalize_contrast(image: np.ndarray) -> np.ndarray:
    """Linearly stretch contrast to [0, 255] range as uint8."""
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        stretched = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        stretched = np.zeros_like(image, dtype=np.uint8)
    return stretched
