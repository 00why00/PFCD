# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, List, Optional, Union, Tuple

import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor, LoRAXFormersAttnProcessor, \
    LoRAAttnProcessor2_0, FusedAttnProcessor2_0
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPVisionModel, GroundingDinoProcessor, \
    GroundingDinoModel

from models.unet_2d_condition import ImageVariationUNet2DConditionModel as UNet2DConditionModel
from utils.training_utils import StableDiffusionXLImageVariationLoraLoaderMixin, \
    StableDiffusionXLImageVariationClassEmbeddingMixin
from utils.training_utils import crop_content_single

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionXLImageVariationPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    StableDiffusionXLImageVariationLoraLoaderMixin,
    StableDiffusionXLImageVariationClassEmbeddingMixin,
):
    r"""
    Pipeline to generate image variations from an input image using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "image_encoder->image_encoder2->unet->vae"
    _optional_components = [
        "feature_extractor",
        "feature_extractor2",
        "image_encoder",
        "image_encoder2",
    ]
    _callback_tensor_inputs = [
        "latents",
        "image_embeds",
        "negative_image_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_image_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModel,
        image_encoder2: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        feature_extractor2: CLIPImageProcessor,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        gd_processor: GroundingDinoProcessor=None,
        gd_model: GroundingDinoModel=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            image_encoder2=image_encoder2,
            feature_extractor=feature_extractor,
            feature_extractor2=feature_extractor2,
            unet=unet,
            scheduler=scheduler,
            gd_processor=gd_processor,
            gd_model=gd_model,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: PIL.Image.Image,
        image_2: Optional[PIL.Image.Image] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_image: Optional[PIL.Image.Image] = None,
        negative_image_2: Optional[PIL.Image.Image] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        pooled_image_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_image_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encode the input image(s) to get the image embeddings and hidden states.
        """
        device = device or self._execution_device
        image = [image] if isinstance(image, PIL.Image.Image) else image

        if image is not None:
            batch_size = len(image)
        else:
            batch_size = image_embeds.shape[0]

        # define feature extractors and image encoders
        feature_extractors = [self.feature_extractor, self.feature_extractor2] if self.feature_extractor2 is not None else [self.feature_extractor2]
        image_encoders = [self.image_encoder, self.image_encoder2] if self.image_encoder is not None else [self.image_encoder2]

        if image_embeds is None:
            image_2 = image_2 or image
            image_2 = [image_2] if isinstance(image_2, PIL.Image.Image) else image_2

        image_embeds_list = []
        images = [image, image_2]
        for image, feature_extractor, image_encoder in zip(images, feature_extractors, image_encoders):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(image.to(device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_image_embeds = image_embeds[0]
            image_embeds = image_embeds.image_embeds
            image_embeds_list.append(image_embeds)

        image_embeds = torch.cat(image_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_image = negative_image is None
        if do_classifier_free_guidance and negative_image_embeds is None and zero_out_negative_image:
            negative_image_embeds = torch.zeros_like(image_embeds)
            negative_pooled_image_embeds = torch.zeros_like(pooled_image_embeds)
        elif do_classifier_free_guidance and negative_image_embeds is None:
            negative_image = negative_image or PIL.Image.new("RGB", image.size)
            negative_image_2 = negative_image_2 or negative_image

            negative_image = batch_size * [negative_image] if isinstance(negative_image, PIL.Image.Image) else negative_image
            negative_image_2 = batch_size * [negative_image_2] if isinstance(negative_image_2, PIL.Image.Image) else negative_image_2

            uncond_images: List[str]
            if image is not None and type(image) is not type(negative_image):
                raise TypeError(
                    f"`negative_image` should be the same type to `image`, but got {type(negative_image)} !="
                    f" {type(image)}."
                )
            elif batch_size != len(negative_image):
                raise ValueError(
                    f"`negative_image`: {negative_image} has batch size {len(negative_image)}, but `image`:"
                    f" {image} has batch size {batch_size}. Please make sure that passed `negative_image` matches"
                    " the batch size of `image`."
                )
            else:
                uncond_images = [negative_image, negative_image_2]

            negative_image_embeds_list = []
            for negative_image, feature_extractor, image_encoder in zip(uncond_images, feature_extractors, image_encoders):
                negative_image = self.feature_extractor(images=negative_image, return_tensors="pt").pixel_values
                negative_image_embeds = image_encoder(negative_image.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_image_embeds = negative_image_embeds[0]
                negative_image_embeds = negative_image_embeds.image_embeds
                negative_image_embeds_list.append(negative_image_embeds)

            negative_image_embeds = torch.cat(negative_image_embeds_list, dim=-1)

        if self.image_encoder2 is not None:
            image_embeds = image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
        else:
            image_embeds = image_embeds.to(dtype=self.unet.dtype, device=device)

        image_embeds = image_embeds.unsqueeze(1)
        negative_image_embeds = negative_image_embeds.unsqueeze(1)
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, num_images_per_prompt, 1)
        image_embeds = image_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_image_embeds.shape[1]

            if self.image_encoder2 is not None:
                negative_image_embeds = negative_image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
            else:
                negative_image_embeds = negative_image_embeds.to(dtype=self.unet.dtype, device=device)

            negative_image_embeds = negative_image_embeds.repeat(1, num_images_per_prompt, 1)
            negative_image_embeds = negative_image_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_image_embeds = pooled_image_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_image_embeds = negative_pooled_image_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return image_embeds, negative_image_embeds, pooled_image_embeds, negative_pooled_image_embeds

    def _encode_image_with_content(
        self,
        image: PIL.Image.Image,
        image_2: Optional[PIL.Image.Image] = None,
        image_info: Optional[dict] = None,
        image_info_2: Optional[dict] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_image: Optional[PIL.Image.Image] = None,
        negative_image_2: Optional[PIL.Image.Image] = None,
        negative_image_info: Optional[dict] = None,
        negative_image_info_2: Optional[dict] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        pooled_image_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_image_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
                Encode the input image(s) to get the image embeddings and hidden states.
                """
        device = device or self._execution_device
        image = [image] if isinstance(image, PIL.Image.Image) else image

        if image is not None:
            batch_size = len(image)
        else:
            batch_size = image_embeds.shape[0]

        assert batch_size == 1, "Batch size should be 1 for content condition"
        content = crop_content_single(image_info)

        # define feature extractors and image encoders
        feature_extractors = [self.feature_extractor, self.feature_extractor2] if self.feature_extractor2 is not None else [self.feature_extractor2]
        image_encoders = [self.image_encoder, self.image_encoder2] if self.image_encoder is not None else [self.image_encoder2]

        if image_embeds is None:
            image_2 = image_2 or image
            image_info_2 = image_info_2 or image_info
            image_2 = [image_2] if isinstance(image_2, PIL.Image.Image) else image_2

        content_2 = crop_content_single(image_info_2)

        image_embeds_list = []
        images = [image, image_2]
        contents = [content, content_2]
        for image, content, feature_extractor, image_encoder in zip(images, contents, feature_extractors, image_encoders):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(image.to(device)).image_embeds
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_image_embeds = image_embeds[0]

            if content:
                content = self.feature_extractor(images=content, return_tensors="pt").pixel_values
                cont_embeds = image_encoder(content.to(device)).image_embeds
                image_embeds = torch.cat([image_embeds, cont_embeds], dim=0)
            image_embeds_list.append(image_embeds)

        image_embeds = torch.cat(image_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_image = negative_image is None
        if do_classifier_free_guidance and negative_image_embeds is None and zero_out_negative_image:
            negative_image_embeds = torch.zeros_like(image_embeds)
            negative_pooled_image_embeds = torch.zeros_like(pooled_image_embeds)
        elif do_classifier_free_guidance and negative_image_embeds is None:
            negative_image = negative_image or PIL.Image.new("RGB", image.size)
            negative_image_2 = negative_image_2 or negative_image

            negative_image = batch_size * [negative_image] if isinstance(negative_image, PIL.Image.Image) else negative_image
            negative_image_2 = batch_size * [negative_image_2] if isinstance(negative_image_2, PIL.Image.Image) else negative_image_2

            negative_content = crop_content_single(negative_image_info)
            negative_content_2 = crop_content_single(negative_image_info_2)

            uncond_images: List[str]
            if image is not None and type(image) is not type(negative_image):
                raise TypeError(
                    f"`negative_image` should be the same type to `image`, but got {type(negative_image)} !="
                    f" {type(image)}."
                )
            elif batch_size != len(negative_image):
                raise ValueError(
                    f"`negative_image`: {negative_image} has batch size {len(negative_image)}, but `image`:"
                    f" {image} has batch size {batch_size}. Please make sure that passed `negative_image` matches"
                    " the batch size of `image`."
                )
            else:
                uncond_images = [negative_image, negative_image_2]
                uncond_contents = [negative_content, negative_content_2]

            negative_image_embeds_list = []
            for negative_image, negative_content, feature_extractor, image_encoder in zip(uncond_images, uncond_contents, feature_extractors, image_encoders):
                negative_image = self.feature_extractor(images=negative_image, return_tensors="pt").pixel_values
                negative_image_embeds = image_encoder(negative_image.to(device)).image_embeds
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_image_embeds = negative_image_embeds[0]

                if content:
                    negative_content = self.feature_extractor(images=negative_content, return_tensors="pt").pixel_values
                    negative_cont_embeds = image_encoder(negative_content.to(device)).image_embeds
                    negative_image_embeds = torch.cat([negative_image_embeds, negative_cont_embeds], dim=0)

                negative_image_embeds_list.append(negative_image_embeds)

            negative_image_embeds = torch.cat(negative_image_embeds_list, dim=-1)

        if self.image_encoder2 is not None:
            image_embeds = image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
        else:
            image_embeds = image_embeds.to(dtype=self.unet.dtype, device=device)

        image_embeds = image_embeds.unsqueeze(0)
        negative_image_embeds = negative_image_embeds.unsqueeze(0)
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, num_images_per_prompt, 1)
        image_embeds = image_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_image_embeds.shape[1]

            if self.image_encoder2 is not None:
                negative_image_embeds = negative_image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
            else:
                negative_image_embeds = negative_image_embeds.to(dtype=self.unet.dtype, device=device)

            negative_image_embeds = negative_image_embeds.repeat(1, num_images_per_prompt, 1)
            negative_image_embeds = negative_image_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_image_embeds = pooled_image_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_image_embeds = negative_pooled_image_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return image_embeds, negative_image_embeds, pooled_image_embeds, negative_pooled_image_embeds

    def _encode_image_with_grounding_dino(
        self,
        image: PIL.Image.Image,
        image_2: Optional[PIL.Image.Image] = None,
        class_label: Optional[str] = None,
        class_label_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_image: Optional[PIL.Image.Image] = None,
        negative_image_2: Optional[PIL.Image.Image] = None,
        negative_class_label: Optional[str] = None,
        negative_class_label_2: Optional[str] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        pooled_image_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_image_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encode the input image(s) and labels with grounding dino to get the image embeddings and hidden states.
        """
        device = device or self._execution_device
        image = [image] if isinstance(image, PIL.Image.Image) else image

        if image is not None:
            batch_size = len(image)
        else:
            batch_size = image_embeds.shape[0]

        # define feature extractors and image encoders
        feature_extractors = [self.feature_extractor, self.feature_extractor2] if self.feature_extractor2 is not None else [self.feature_extractor2]
        image_encoders = [self.image_encoder, self.image_encoder2] if self.image_encoder is not None else [self.image_encoder2]

        if image_embeds is None:
            image_2 = image_2 or image
            class_label_2 = class_label_2 or class_label
            image_2 = [image_2] if isinstance(image_2, PIL.Image.Image) else image_2
            class_label_2 = [class_label_2] if isinstance(class_label_2, str) else class_label_2

        image_embeds_list = []
        images = [image, image_2]
        class_labels = [class_label, class_label_2]
        for _image, feature_extractor, image_encoder in zip(images, feature_extractors, image_encoders):
            _image = self.feature_extractor(images=_image, return_tensors="pt").pixel_values
            image_embeds = image_encoder(_image.to(device), output_hidden_states=True).image_embeds
            image_embeds_list.append(image_embeds)
        image_embeds = torch.cat(image_embeds_list, dim=-1)

        for _image, class_label in zip(images, class_labels):
            gd_inputs = self.gd_processor(images=_image, text=class_label, return_tensors="pt").to(self.gd_model.device)
            pooled_image_embeds = self.gd_model(**gd_inputs).last_hidden_state[:, -5:, :].flatten()

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_image = negative_image is None
        if do_classifier_free_guidance and negative_image_embeds is None and zero_out_negative_image:
            negative_image_embeds = torch.zeros_like(image_embeds)
            negative_pooled_image_embeds = torch.zeros_like(pooled_image_embeds)
        elif do_classifier_free_guidance and negative_image_embeds is None:
            negative_image = negative_image or PIL.Image.new("RGB", image.size)
            negative_class_label = negative_class_label or ""
            negative_image_2 = negative_image_2 or negative_image
            negative_class_label_2 = negative_class_label_2 or negative_class_label

            negative_image = batch_size * [negative_image] if isinstance(negative_image, PIL.Image.Image) else negative_image
            negative_class_label = batch_size * [negative_class_label] if isinstance(negative_class_label, str) else negative_class_label
            negative_image_2 = batch_size * [negative_image_2] if isinstance(negative_image_2, PIL.Image.Image) else negative_image_2
            negative_class_label_2 = batch_size * [negative_class_label_2] if isinstance(negative_class_label_2, str) else negative_class_label_2

            uncond_images: List[str]
            if image is not None and type(image) is not type(negative_image):
                raise TypeError(
                    f"`negative_image` should be the same type to `image`, but got {type(negative_image)} !="
                    f" {type(image)}."
                )
            elif batch_size != len(negative_image):
                raise ValueError(
                    f"`negative_image`: {negative_image} has batch size {len(negative_image)}, but `image`:"
                    f" {image} has batch size {batch_size}. Please make sure that passed `negative_image` matches"
                    " the batch size of `image`."
                )
            else:
                uncond_images = [negative_image, negative_image_2]
                uncond_class_labels = [negative_class_label, negative_class_label_2]

            negative_image_embeds_list = []
            for _negative_image, feature_extractor, image_encoder in zip(uncond_images, feature_extractors, image_encoders):
                _negative_image = self.feature_extractor(images=_negative_image, return_tensors="pt").pixel_values
                negative_image_embeds = image_encoder(_negative_image.to(device), output_hidden_states=True).image_embeds
                negative_image_embeds_list.append(negative_image_embeds)
            negative_image_embeds = torch.cat(negative_image_embeds_list, dim=-1)

            for _negative_image, _negative_class_label in zip(uncond_images, uncond_class_labels):
                gd_inputs = self.gd_processor(images=_negative_image, text=_negative_class_label, return_tensors="pt").to(self.gd_model.device)
                negative_pooled_image_embeds = self.gd_model(**gd_inputs).last_hidden_state[:, -5:, :].flatten()

        if self.image_encoder2 is not None:
            image_embeds = image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
        else:
            image_embeds = image_embeds.to(dtype=self.unet.dtype, device=device)

        image_embeds = image_embeds.unsqueeze(1)
        negative_image_embeds = negative_image_embeds.unsqueeze(1)
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, num_images_per_prompt, 1)
        image_embeds = image_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_image_embeds.shape[1]

            if self.image_encoder2 is not None:
                negative_image_embeds = negative_image_embeds.to(dtype=self.image_encoder2.dtype, device=device)
            else:
                negative_image_embeds = negative_image_embeds.to(dtype=self.unet.dtype, device=device)

            negative_image_embeds = negative_image_embeds.repeat(1, num_images_per_prompt, 1)
            negative_image_embeds = negative_image_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_image_embeds = pooled_image_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_image_embeds = negative_pooled_image_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return image_embeds, negative_image_embeds, pooled_image_embeds, negative_pooled_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        image_2: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        class_label: Optional[Union[str, List[str]]] = None,
        class_label_2: Optional[Union[str, List[str]]] = None,
        use_content: bool = False,
        image_info: Optional[dict] = None,
        image_info_2: Optional[dict] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        negative_image_2: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        negative_class_label: Optional[Union[str, List[str]]] = None,
        negative_class_label_2: Optional[Union[str, List[str]]] = None,
        negative_image_info: Optional[dict] = None,
        negative_image_info_2: Optional[dict] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        class_labels: Optional[List] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        pooled_image_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width, callback_steps)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input image
        arg_dict = {
            "image": image,
            "image_2": image_2,
            "device": device,
            "num_images_per_prompt": num_images_per_prompt,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "negative_image": negative_image,
            "negative_image_2": negative_image_2,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "pooled_image_embeds": pooled_image_embeds,
            "negative_pooled_image_embeds": negative_pooled_image_embeds,
        }
        if use_content:
            assert image_info is not None, "Please provide `image_info` for content-based guidance."
            arg_dict.update({
                "image_info": image_info,
                "image_info_2": image_info_2,
                "negative_image_info": negative_image_info,
                "negative_image_info_2": negative_image_info_2,
            })
            encode_func = self._encode_image_with_content
        elif self.gd_processor is not None and self.gd_model is not None:
            arg_dict.update({
                "class_label": class_label,
                "class_label_2": class_label_2,
                "negative_class_label": negative_class_label,
                "negative_class_label_2": negative_class_label_2,
            })
            encode_func = self._encode_image_with_grounding_dino
        else:
            encode_func = self._encode_image

        (
            image_embeds, negative_image_embeds,
            pooled_image_embeds, negative_pooled_image_embeds,
        ) = encode_func(**arg_dict)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_image_embeds
        if self.image_encoder2 is None:
            image_encoder_projection_dim = int(pooled_image_embeds.shape[-1])
        else:
            image_encoder_projection_dim = self.image_encoder2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=image_embeds.dtype,
            text_encoder_projection_dim=image_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=image_embeds.dtype,
                text_encoder_projection_dim=image_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_image_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            if class_labels is not None:
                class_labels.insert(0, [])

        image_embeds = image_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeds,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
