from typing import Optional

import torch
from diffusers import UNet2DConditionModel


class ImageVariationUNet2DConditionModel(
    UNet2DConditionModel
):
    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when class_embedding is not None")

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb