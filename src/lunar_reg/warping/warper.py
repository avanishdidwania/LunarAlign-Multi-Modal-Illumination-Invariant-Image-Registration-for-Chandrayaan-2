import cv2
import numpy as np

class ImageWarper:
    """
    Image warping and blending module.
    Transforms source images into reference coordinates and blends them.
    """

    def warp(self, image: np.ndarray, matrix: np.ndarray,
             reference_shape: tuple[int, int],
             interpolation: str = "bilinear",
             border_value: float = 0.0) -> np.ndarray:
        """
        Warp source image to reference coordinate frame using the 3x3 transformation matrix.
        reference_shape: (height, width)
        """
        h_ref, w_ref = reference_shape
        
        # Map interpolation type to OpenCV flags
        if interpolation == "nearest":
            flags = cv2.INTER_NEAREST
        elif interpolation == "bicubic":
            flags = cv2.INTER_CUBIC
        else:
            flags = cv2.INTER_LINEAR
            
        warped = cv2.warpPerspective(
            image,
            matrix,
            (w_ref, h_ref),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        return warped

    def blend(self, warped_source: np.ndarray,
              reference_image: np.ndarray,
              blend_mode: str = "overlay") -> np.ndarray:
        """
        Blend the warped source image with the reference image.
        Supported modes:
        - "overlay": overlays warped_source on top of reference_image where warped_source is non-zero.
        - "average": averages the two images in overlapping regions.
        - "difference": absolute difference of the two images.
        """
        # Ensure identical shapes
        assert warped_source.shape == reference_image.shape, "Images must have the same shape to blend."
        
        # Create mask of valid warped pixels (where it is non-zero)
        valid_mask = warped_source > 0
        
        if blend_mode == "overlay":
            blended = reference_image.copy()
            blended[valid_mask] = warped_source[valid_mask]
            return blended
            
        elif blend_mode == "average":
            blended = reference_image.copy().astype(np.float32)
            # Where both are valid, average them
            both_valid = (warped_source > 0) & (reference_image > 0)
            blended[both_valid] = (warped_source[both_valid].astype(np.float32) + reference_image[both_valid].astype(np.float32)) / 2.0
            # Where only warped_source is valid, use warped_source
            only_warped = (warped_source > 0) & (reference_image == 0)
            blended[only_warped] = warped_source[only_warped].astype(np.float32)
            return blended.astype(reference_image.dtype)
            
        elif blend_mode == "difference":
            # Absolute difference in overlapping region
            overlap = (warped_source > 0) & (reference_image > 0)
            diff = np.zeros_like(reference_image)
            # Avoid underflow for uint8
            if reference_image.dtype == np.uint8:
                diff[overlap] = cv2.absdiff(warped_source, reference_image)[overlap]
            else:
                diff[overlap] = np.abs(warped_source[overlap].astype(np.float32) - reference_image[overlap].astype(np.float32)).astype(reference_image.dtype)
            return diff
            
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
