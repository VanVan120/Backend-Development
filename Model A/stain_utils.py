import numpy as np
import cv2

class MacenkoNormalizer:
    def __init__(self):
        # Standard Reference H&E OD matrix (from a "perfect" slide)
        # These values are empirically derived from a standard dataset (e.g., TCGA)
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # Calculate optical density
        I = (I.astype(np.float32) + 1) / Io
        OD = -np.log(I)
        
        # Remove data with too little optical density (background)
        ODhat = OD.reshape((-1, 3))
        mask = np.all(ODhat < beta, axis=1)
        return OD, ODhat, mask

    def normalize(self, img, Io=240, beta=0.15):
        """
        Normalize the staining of an H&E image.
        img: Input image (RGB, uint8, numpy array)
        """
        h, w, c = img.shape
        
        # 1. Convert to Optical Density
        OD, ODhat, mask = self.__convert_rgb2od(img, Io, beta)
        
        # If image is mostly background (white), return as is
        if np.sum(~mask) < 10:
            return img

        # 2. Calculate Eigenvectors of the OD matrix (Stain Matrix)
        OD_tissue = ODhat[~mask]
        
        # Covariance matrix
        _, V = np.linalg.eigh(np.cov(OD_tissue, rowvar=False))
        
        # The two principle eigenvectors
        Vec = V[:, 1:3]
        
        # Project data onto the plane spanned by the eigenvectors
        That = np.dot(OD_tissue, Vec)
        
        # 3. Determine the robust extremes of the angle
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, 1)
        maxPhi = np.percentile(phi, 99)
        
        vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # Heuristic: H usually has a larger R component than E? 
        # Standard sorting: vMin is usually H, vMax is E
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T
            
        # 4. Determine concentrations of the individual stains
        # Y = OD = C * HE.T  => C = Y * pinv(HE.T)
        Y = OD.reshape((-1, 3)).T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        
        # 5. Normalize concentrations
        # Scale concentrations to match the reference max concentrations
        maxC = np.percentile(C, 99, axis=1)
        
        # Avoid division by zero
        maxC[maxC == 0] = 0.001
        
        C_norm = C * (self.maxCRef[:, np.newaxis] / maxC[:, np.newaxis])
        
        # 6. Recreate the image using the Reference Stain Matrix
        # OD_norm = C_norm.T * HERef.T
        OD_norm = np.dot(self.HERef, C_norm)
        
        # Convert back to RGB
        I_norm = Io * np.exp(-OD_norm)
        I_norm = np.clip(I_norm, 0, 255).astype(np.uint8)
        I_norm = I_norm.T.reshape(h, w, c)
        
        return I_norm
