import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from pipelines.pipeline_stable_xl_image_variation import StableDiffusionXLImageVariationPipeline

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
image_encoder2 = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
feature_extractor2 = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

sd_pipe = StableDiffusionXLImageVariationPipeline(
    vae=vae,
    image_encoder=image_encoder,
    image_encoder2=image_encoder2,
    feature_extractor=feature_extractor,
    feature_extractor2=feature_extractor2,
    unet=unet,
    scheduler=scheduler,
)
sd_pipe = sd_pipe.to('cuda', dtype=torch.float16)

path = "data/COCO/train2017/000000000009.jpg"
init_image = load_image(path)

out = sd_pipe(init_image, guidance_scale=3, generator=torch.Generator().manual_seed(0))
out["images"][0].save("result_xl.jpg")

sd_pipe.save_pretrained("saved_pipeline/stable-xl-image-variation")
