from albumentations import (
    Compose,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    CoarseDropout,
    Sequential,
    OneOf,
)
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class Transforms:
    def __init__(self, means, stds, train=True):
        if train:
            self.transformations = Compose(
                [
                    OneOf([
                        Sequential([
                            PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # padding of 4 on each side of 32x32 image
                            RandomCrop(height=32, width=32, always_apply=True),
                        ], p=0.5),
                        Sequential([
                            CoarseDropout(max_height=16, max_width=16, min_height=16, min_width=16, min_holes=1, max_holes=1, fill_value=means, always_apply=True),
                        ], p=0.5)
                    ], p=1),  # Always apply at least one of the above transformations.
                    Normalize(mean=means, std=stds, always_apply=True),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = Compose(
                [
                    Normalize(mean=means, std=stds, always_apply=True),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image=np.array(img))["image"]
