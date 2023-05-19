import cv2
import random
from albumentations.core.transforms_interface import ImageOnlyTransform

class JpegCompression(ImageOnlyTransform):
    def __init__(self, quality_lower=99, quality_upper=100, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, img, **params):
        # Randomly choose a quality for jpeg compression
        quality = random.randint(self.quality_lower, self.quality_upper)

        # Encode image to jpeg
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_img = cv2.imencode('.jpg', img, encode_param)
        
        # If successful, decode the image back to numpy array
        if result:
            img = cv2.imdecode(encoded_img, 1)
        return img
    
    @classmethod
    def get_transform_init_args_names(cls):
        return ("quality_lower", "quality_upper", "always_apply", "p")