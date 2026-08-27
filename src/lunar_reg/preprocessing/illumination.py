import cv2
import numpy as np

class IlluminationNormalizer:
    """Illumination-invariant preprocessing for lunar imagery."""

    def phase_congruency(self, image: np.ndarray, n_scales: int = 4,
                         n_orientations: int = 6) -> np.ndarray:
        """
        Compute phase congruency map - illumination invariant feature image.
        Uses log-Gabor filters across multiple scales and orientations.
        Returns a map where high values indicate features regardless of illumination.
        """
        # Ensure image is 2D float32
        img = image.astype(np.float32)
        rows, cols = img.shape
        
        # Compute FFT
        img_fft = np.fft.fft2(img)
        
        # Setup X and Y frequency coordinates [-0.5, 0.5]
        x = np.linspace(-0.5, 0.5 - 1.0/cols, cols)
        y = np.linspace(-0.5, 0.5 - 1.0/rows, rows)
        xv, yv = np.meshgrid(x, y)
        
        # Shift to match FFT layout (origin at top-left)
        xv = np.fft.ifftshift(xv)
        yv = np.fft.ifftshift(yv)
        
        # Calculate radius and theta for each grid point
        radius = np.sqrt(xv**2 + yv**2)
        radius[0, 0] = 1.0  # Avoid zero division at origin
        
        theta = np.arctan2(-yv, xv)
        # Convert theta to range [0, pi]
        theta[theta < 0] += np.pi
        
        # Log-Gabor parameters
        min_wavelength = 3.0
        mult = 2.1
        sigma_on_f = 0.55
        d_theta = np.pi / n_orientations
        theta_sigma = d_theta * 1.2 / 2.0
        
        total_energy = np.zeros_like(img)
        total_amplitude = np.zeros_like(img)
        
        for o in range(n_orientations):
            o_angle = o * d_theta
            
            # Compute angular filter component wrapping around pi
            diff = np.abs(theta - o_angle)
            diff = np.minimum(diff, np.pi - diff)
            angular_filter = np.exp(-(diff**2) / (2 * theta_sigma**2))
            
            sum_even = np.zeros_like(img)
            sum_odd = np.zeros_like(img)
            sum_amp = np.zeros_like(img)
            
            amp_list = []
            
            for s in range(n_scales):
                wavelength = min_wavelength * (mult**s)
                fo = 1.0 / wavelength
                
                # Radial filter: log-Gabor
                radial_filter = np.exp(-((np.log(radius / fo))**2) / (2 * (np.log(sigma_on_f))**2))
                radial_filter[0, 0] = 0.0  # Zero DC component
                
                gabor_filter = radial_filter * angular_filter
                
                # Filter in frequency domain
                filtered_fft = img_fft * gabor_filter
                filtered = np.fft.ifft2(filtered_fft)
                
                even = np.real(filtered)
                odd = np.imag(filtered)
                amp = np.sqrt(even**2 + odd**2)
                
                sum_even += even
                sum_odd += odd
                sum_amp += amp
                amp_list.append(amp)
                
            energy = np.sqrt(sum_even**2 + sum_odd**2)
            
            # Noise estimation: median of the high frequency filter amplitude
            # as high frequencies are dominated by noise.
            noise_est = np.median(amp_list[0]) * 2.0
            
            energy_clean = np.maximum(0.0, energy - noise_est)
            
            total_energy += energy_clean
            total_amplitude += sum_amp
            
        epsilon = 1e-4
        pc = total_energy / (total_amplitude + epsilon)
        return np.clip(pc, 0.0, 1.0)

    def phase_congruency_tiled(self, image: np.ndarray, n_scales: int = 4,
                               n_orientations: int = 6, tile_size: int = 2048,
                               overlap: int = 128) -> np.ndarray:
        """
        Compute phase congruency in tiles to prevent Out-Of-Memory (OOM) errors
        on extremely large satellite tracks (e.g. 4000 x 200,000).
        """
        rows, cols = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        # Calculate step sizes
        y_step = tile_size - 2 * overlap
        x_step = tile_size - 2 * overlap
        
        for y in range(0, rows, y_step):
            y_start = max(0, y - overlap)
            y_end = min(rows, y + tile_size - overlap)
            
            for x in range(0, cols, x_step):
                x_start = max(0, x - overlap)
                x_end = min(cols, x + tile_size - overlap)
                
                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]
                if tile.size == 0:
                    continue
                    
                # Compute PC for this tile
                tile_pc = self.phase_congruency(tile, n_scales=n_scales, n_orientations=n_orientations)
                
                # Determine output target region (non-overlapping region)
                out_y_start = y
                out_y_end = min(rows, y + y_step)
                out_x_start = x
                out_x_end = min(cols, x + x_step)
                
                # Crop corresponding region from tile_pc
                tile_y_start = out_y_start - y_start
                tile_y_end = out_y_end - y_start
                tile_x_start = out_x_start - x_start
                tile_x_end = out_x_end - x_start
                
                result[out_y_start:out_y_end, out_x_start:out_x_end] = tile_pc[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                
        return result

    def local_normalized_mean_subtraction(self, image: np.ndarray,
                                          kernel_size: int = 31) -> np.ndarray:
        """
        Subtract local mean and divide by local standard deviation.
        Removes low-frequency illumination gradients while preserving texture.
        """
        img = image.astype(np.float32)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        mean = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
        mean_sq = cv2.boxFilter(img**2, -1, (kernel_size, kernel_size))
        variance = np.maximum(0.0, mean_sq - mean**2)
        std = np.sqrt(variance)
        
        epsilon = 1e-4
        normalized = (img - mean) / (std + epsilon)
        return normalized

    def gradient_orientation_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient orientation map (0 to 2*pi).
        Gradient direction is illumination-invariant as shadows shift
        intensity but not edge direction.
        """
        img = image.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        
        ori = np.arctan2(gy, gx)
        ori[ori < 0] += 2 * np.pi
        return ori

    def clahe(self, image: np.ndarray, clip_limit: float = 2.0,
              tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization."""
        if image.dtype != np.uint8:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                img_uint8 = ((image - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
            else:
                img_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            img_uint8 = image
            
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        equalized = clahe_obj.apply(img_uint8)
        return equalized

    def normalize(self, image: np.ndarray, method: str = "phase_congruency") -> np.ndarray:
        """Apply selected illumination normalization method."""
        if method == "phase_congruency":
            if image.shape[0] > 2048 or image.shape[1] > 2048:
                return self.phase_congruency_tiled(image)
            return self.phase_congruency(image)
        elif method == "clahe":
            return self.clahe(image)
        elif method == "gradient":
            return self.gradient_orientation_map(image)
        elif method == "lnms":
            return self.local_normalized_mean_subtraction(image)
        else:
            raise ValueError(f"Unknown illumination normalization method: {method}")
