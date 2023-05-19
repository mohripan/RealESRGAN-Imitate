import torch
import albumentations as A
import torch.nn.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform

# class SincFilter:
#     def __init__(self, size):
#         self.size = size
#         self.filter = self.create_sinc_filter()

#     def create_sinc_filter(self):
#         k = torch.linspace(-1, 1, self.size)
#         x = torch.outer(torch.sinc(k), torch.sinc(k))
#         return x

#     def __call__(self, img):
#         img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
#         sinc_filter = self.filter.unsqueeze(0).unsqueeze(0)
#         return F.conv2d(img, sinc_filter, padding=self.size//2).squeeze().numpy()

class SincFilterTransform(ImageOnlyTransform):
    def __init__(self, size=5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        return self.sinc_filter(img, self.size)

    @staticmethod
    def sinc_filter(img, size):
        class SincFilter:
            def __init__(self, size):
                self.size = size
                self.filter = self.create_sinc_filter()

            def create_sinc_filter(self):
                k = torch.linspace(-1, 1, self.size)
                x = torch.outer(torch.sinc(k), torch.sinc(k))
                return x

            def __call__(self, img):
                img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
                sinc_filter = self.filter.unsqueeze(0).unsqueeze(0)
                filtered_image = F.conv2d(img, sinc_filter, padding=self.size//2)
                return filtered_image.squeeze().numpy()

        sinc_filter = SincFilter(size)
        return sinc_filter(img)

    @classmethod
    def get_transform_init_args_names(cls):
        return ("size", "always_apply", "p")