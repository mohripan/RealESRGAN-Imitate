import torch
import albumentations as A
import torch.nn.functional as F

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
        return F.conv2d(img, sinc_filter, padding=self.size//2).squeeze().numpy()

class SincFilterTransform(A.BasicTransform):
    def __init__(self, size, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.sinc_filter = SincFilter(size)

    def apply(self, image, **params):
        return self.sinc_filter(image)

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('size',)