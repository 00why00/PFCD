import os

import cv2
import lpips
from pytorch_image_generation_metrics import get_inception_score_and_fid, ImageDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# IS and FID
input_path = "data/COCO-YOLO/images/content-counting_loss-th0.1-step1000-scale0.5"
val_path = "data/COCO-YOLO/images/val2017"
fid_ref = "data/COCO-YOLO/images/val2017.npz"
(IS, IS_std), FID = get_inception_score_and_fid(
    DataLoader(ImageDataset(
        input_path,
        transform=Compose([
            Resize((512, 512)),
            ToTensor()
        ])
    ), batch_size=50),
    fid_ref,
    use_torch=True
)

# LPIPS
LPIPS = 0
loss_fn = lpips.LPIPS().cuda()
for image in os.listdir(input_path):
    input_image_path = os.path.join(input_path, image)
    val_image_path = os.path.join(val_path, image)
    assert os.path.exists(input_image_path) and os.path.exists(val_image_path)
    img0 = lpips.im2tensor(cv2.resize(lpips.load_image(input_image_path), (512, 512))).cuda()
    img1 = lpips.im2tensor(cv2.resize(lpips.load_image(val_image_path), (512, 512))).cuda()
    loss = loss_fn(img0, img1)
    LPIPS += loss.item()
LPIPS /= len(os.listdir(input_path))

print(IS, IS_std, FID, LPIPS)
