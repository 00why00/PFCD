import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from transformers import AutoModelForZeroShotObjectDetection

from grounded_sam.processing_grounding_dino import GroundingDinoProcessor


class GroundingDino(nn.Module):
    def __init__(self, model_name, device):
        super(GroundingDino, self).__init__()
        self.processor = GroundingDinoProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, local_files_only=True)
        self.model.to(device, dtype=torch.float32)

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        self.std = [0.229, 0.224, 0.225]
        self.mean = [0.485, 0.456, 0.406]
        self.short_size = 800
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.short_size, self.short_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, images, labels, label_count, label_index, counting_loss_threshold, counting_loss_scale):
        images = torch.stack([self.transforms(image) for image in images])
        inputs = self.processor(images=images,
                                text=labels,
                                padding=True,
                                return_tensors="pt").to(self.model.device)

        with torch.autocast(device_type="cuda", cache_enabled=False):
            logits = self.model(**inputs).logits

        logits = torch.sigmoid(logits)

        counting_loss = images.new_zeros(())
        for (logit, count, lb_index) in zip(logits, label_count, label_index):
            single_counting_loss = images.new_zeros(())
            if len(count) == 0:
                continue
            for (c, li) in zip(count, lb_index):
                if isinstance(li, int):
                    score = logit[:, li]
                else:
                    score = torch.cat([logit[:, l] for l in li], dim=-1)
                instance_score = F.relu(counting_loss_threshold - torch.topk(score, k=c, dim=-1)[0])
                single_counting_loss += torch.sum(instance_score)
            counting_loss += single_counting_loss / sum(count) * counting_loss_scale
        return counting_loss