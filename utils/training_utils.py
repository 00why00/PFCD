import copy
import inspect
import os
from random import randrange
from typing import Optional, Tuple, Union, Dict, Callable, List

import numpy as np
import torch
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import SchedulerMixin
from diffusers.utils import USE_PEFT_BACKEND, convert_state_dict_to_diffusers, convert_state_dict_to_peft, \
    get_peft_kwargs, is_peft_version, get_adapter_name, scale_lora_layers, recurse_remove_peft_layers, \
    set_weights_and_activate_adapters, set_adapter_layers, delete_adapter_layers
from torch import nn
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
from transformers.utils import logging

logger = logging.get_logger(__name__)


def compute_dream_and_update_latents(
    unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.
    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.
    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    with torch.no_grad():
        pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if noise_scheduler.config.prediction_type == "epsilon":
        predicted_noise = pred
        delta_noise = (noise - predicted_noise).detach()
        delta_noise.mul_(dream_lambda)
        noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        target = target.add(delta_noise)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    return noisy_latents, target

def crop_content(examples, pad=True):
    crop_image_list = []
    for image, width, height, instances in zip(examples['image'], examples['width'], examples['height'], examples['instances']):
        crop_images = []
        if len(instances) == 0:
            crop_image_list.append([])
            continue
        for idx, instance in enumerate(instances):
            bbox = instance['bbox']
            if pad:
                crop_img = image.crop((
                    max(bbox[0] - 50, 0),
                    max(bbox[1] - 50, 0),
                    min(bbox[0] + bbox[2] + 50, width),
                    min(bbox[1] + bbox[3] + 50, height)
                ))
            else:
                crop_img = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            crop_images.append(crop_img)
        crop_image_list.append(crop_images)
    return crop_image_list

def random_crop_content(examples):
    crop_image_list = []
    for image, width, height, instances in zip(examples['image'], examples['width'], examples['height'], examples['instances']):
        crop_images = []
        matrix_list = [min(width, height) // 2, min(width, height) // 3, min(width, height) // 4]
        for length in matrix_list:
            for _ in range(3):
                w1 = randrange(0, width - length)
                h1 = randrange(0, height - length)
                crop_img = image.crop((w1, h1, w1 + length, h1 + length))
                crop_images.append(crop_img)
        crop_image_list.append(crop_images)
    return crop_image_list

def bbox_jitter_content(examples, jitter_ratio=0.1, pad=True):
    crop_image_list = []
    for image, width, height, instances in zip(examples['image'], examples['width'], examples['height'], examples['instances']):
        crop_images = []
        if len(instances) == 0:
            crop_image_list.append([])
            continue
        for idx, instance in enumerate(instances):
            bbox = instance['bbox']
            scale = np.random.uniform(-jitter_ratio, jitter_ratio, 4)
            if pad:
                x1 = max(bbox[0] - 50, 0)
                y1 = max(bbox[1] - 50, 0)
                x2 = min(bbox[0] + bbox[2] + 50, width)
                y2 = min(bbox[1] + bbox[3] + 50, height)
            else:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
            # apply bbox jitter
            jitter_x1 = x1 + scale[0] * (x2 - x1)
            jitter_y1 = y1 + scale[1] * (y2 - y1)
            jitter_x2 = x2 + scale[2] * (x2 - x1)
            jitter_y2 = y2 + scale[3] * (y2 - y1)
            # sanity check
            jitter_x1 = max(0, min(width, jitter_x1))
            jitter_y1 = max(0, min(height, jitter_y1))
            jitter_x2 = max(0, min(width, jitter_x2))
            jitter_y2 = max(0, min(height, jitter_y2))
            # crop image
            crop_img = image.crop((jitter_x1, jitter_y1, jitter_x2, jitter_y2))
            crop_images.append(crop_img)
        crop_image_list.append(crop_images)
    return crop_image_list

def crop_content_single(image_info, pad=True):
    image = image_info['image']
    width = image_info['width']
    height = image_info['height']
    instances = image_info['instances']
    crop_images = []
    if len(instances) == 0:
        return []
    for idx, instance in enumerate(instances):
        bbox = instance['bbox']
        if pad:
            crop_img = image.crop((
                max(bbox[0] - 50, 0),
                max(bbox[1] - 50, 0),
                min(bbox[0] + bbox[2] + 50, width),
                min(bbox[1] + bbox[3] + 50, height)
            ))
        else:
            crop_img = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        crop_images.append(crop_img)
    return crop_images

def image_encoder_attn_modules(image_encoder):
    attn_modules = []

    if isinstance(image_encoder, (CLIPVisionModel, CLIPVisionModelWithProjection)):
        for i, layer in enumerate(image_encoder.vision_model.encoder.layers):
            name = f"vision_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {image_encoder.__class__.__name__}")

    return attn_modules

def image_encoder_mlp_modules(image_encoder):
    mlp_modules = []

    if isinstance(image_encoder, (CLIPVisionModel, CLIPVisionModelWithProjection)):
        for i, layer in enumerate(image_encoder.vision_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f"vision_model.encoder.layers.{i}.mlp"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {image_encoder.__class__.__name__}")

    return mlp_modules

class StableDiffusionXLImageVariationLoraLoaderMixin(StableDiffusionXLLoraLoaderMixin):
    """This class overrides `StableDiffusionXLLoraLoaderMixin` with LoRA loading/saving code that's specific to ImageVariation"""

    image_encoder_name = "image_encoder"

    # Override to properly handle the loading and unloading of the image encoder.
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.image_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_image_encoder`] for more details on how the state dict is loaded
        into `self.image_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_unet(
            state_dict, network_alphas=network_alphas, unet=self.unet, adapter_name=adapter_name, _pipeline=self
        )
        image_encoder_state_dict = {k: v for k, v in state_dict.items() if "image_encoder." in k}
        if len(image_encoder_state_dict) > 0:
            self.load_lora_into_image_encoder(
                image_encoder_state_dict,
                network_alphas=network_alphas,
                image_encoder=self.image_encoder,
                prefix="image_encoder",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
            )

        image_encoder2_state_dict = {k: v for k, v in state_dict.items() if "image_encoder2." in k}
        if len(image_encoder2_state_dict) > 0:
            self.load_lora_into_image_encoder(
                image_encoder2_state_dict,
                network_alphas=network_alphas,
                image_encoder=self.image_encoder2,
                prefix="image_encoder2",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
            )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        image_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        image_encoder2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and image encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            image_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `image_encoder`. Must explicitly pass the image
                encoder LoRA state dict because it comes from 🤗 Transformers.
            image_encoder2_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `image_encoder2`. Must explicitly pass the image
                encoder LoRA state dict because it comes from 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            weight_name (`str`, *optional*):
                Name of the weight file to save. If not provided, the weight file will be named `lora_weights.pt`.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or image_encoder_lora_layers or image_encoder2_lora_layers):
            raise ValueError(
                "You must pass at least one of `unet_lora_layers`, `image_encoder_lora_layers` or `image_encoder2_lora_layers`."
            )

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if image_encoder_lora_layers:
            state_dict.update(pack_weights(image_encoder_lora_layers, "image_encoder"))

        if image_encoder2_lora_layers:
            state_dict.update(pack_weights(image_encoder2_lora_layers, "image_encoder2"))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    @classmethod
    def load_lora_into_image_encoder(
        cls,
        state_dict,
        network_alphas,
        image_encoder,
        prefix=None,
        lora_scale=1.0,
        adapter_name=None,
        _pipeline=None,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `image_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key should be prefixed with an
                additional `image_encoder` to distinguish between unet lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            image_encoder (`CLIPVisionModel`):
                The image encoder model to load the LoRA layers into.
            prefix (`str`):
                Expected prefix of the `image_encoder` in the `state_dict`.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft import LoraConfig

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.image_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        prefix = cls.image_encoder_name if prefix is None else prefix

        # Safe prefix to check with.
        if any(cls.image_encoder_name in key for key in keys):
            # Load the layers corresponding to image encoder and make necessary adjustments.
            image_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
            image_encoder_lora_state_dict = {
                k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in image_encoder_keys
            }

            if len(image_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {prefix}.")
                rank = {}
                image_encoder_lora_state_dict = convert_state_dict_to_diffusers(image_encoder_lora_state_dict)

                # convert state dict
                image_encoder_lora_state_dict = convert_state_dict_to_peft(image_encoder_lora_state_dict)

                for name, _ in image_encoder_attn_modules(image_encoder):
                    rank_key = f"{name}.out_proj.lora_B.weight"
                    rank[rank_key] = image_encoder_lora_state_dict[rank_key].shape[1]

                patch_mlp = any(".mlp." in key for key in image_encoder_lora_state_dict.keys())
                if patch_mlp:
                    for name, _ in image_encoder_mlp_modules(image_encoder):
                        rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                        rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                        rank[rank_key_fc1] = image_encoder_lora_state_dict[rank_key_fc1].shape[1]
                        rank[rank_key_fc2] = image_encoder_lora_state_dict[rank_key_fc2].shape[1]

                if network_alphas is not None:
                    alpha_keys = [
                        k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix
                    ]
                    network_alphas = {
                        k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                lora_config_kwargs = get_peft_kwargs(rank, network_alphas, image_encoder_lora_state_dict, is_unet=False)
                if "use_dora" in lora_config_kwargs:
                    if lora_config_kwargs["use_dora"]:
                        if is_peft_version("<", "0.9.0"):
                            raise ValueError(
                                "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                            )
                    else:
                        if is_peft_version("<", "0.9.0"):
                            lora_config_kwargs.pop("use_dora")
                lora_config = LoraConfig(**lora_config_kwargs)

                # adapter_name
                if adapter_name is None:
                    adapter_name = get_adapter_name(image_encoder)

                is_model_cpu_offload, is_sequential_cpu_offload = cls._optionally_disable_offloading(_pipeline)

                # inject LoRA layers and load the state dict
                # in transformers we automatically check whether the adapter name is already in use or not
                image_encoder.load_adapter(
                    adapter_name=adapter_name,
                    adapter_state_dict=image_encoder_lora_state_dict,
                    peft_config=lora_config,
                )

                # scale LoRA layers with `lora_scale`
                scale_lora_layers(image_encoder, weight=lora_scale)

                image_encoder.to(device=image_encoder.device, dtype=image_encoder.dtype)

                # Offload back.
                if is_model_cpu_offload:
                    _pipeline.enable_model_cpu_offload()
                elif is_sequential_cpu_offload:
                    _pipeline.enable_sequential_cpu_offload()
                # Unsafe code />

    def _remove_image_encoder_monkey_patch(self):
        recurse_remove_peft_layers(self.image_encoder)
        if getattr(self.image_encoder, "peft_config", None) is not None:
            del self.image_encoder.peft_config
            self.image_encoder._hf_peft_config_loaded = None

        recurse_remove_peft_layers(self.image_encoder2)
        if getattr(self.image_encoder2, "peft_config", None) is not None:
            del self.image_encoder2.peft_config
            self.image_encoder2._hf_peft_config_loaded = None

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet.unload_lora()

        # Safe to call the following regardless of LoRA.
        self._remove_image_encoder_monkey_patch()

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_image_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            fuse_unet (`bool`, defaults to `True`): Whether to fuse the UNet LoRA parameters.
            fuse_image_encoder (`bool`, defaults to `True`):
                Whether to fuse the image encoder LoRA parameters. If the image encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        if fuse_unet or fuse_image_encoder:
            self.num_fused_loras += 1
            if self.num_fused_loras > 1:
                logger.warning(
                    "The current API is supported for operating with a single LoRA file. You are trying to load and fuse more than one LoRA which is not well-supported.",
                )

        if fuse_unet:
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            unet.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=adapter_names)

        def fuse_image_encoder_lora(image_encoder, lora_scale=1.0, safe_fusing=False, adapter_names=None):
            merge_kwargs = {"safe_merge": safe_fusing}

            for module in image_encoder.modules():
                if isinstance(module, BaseTunerLayer):
                    if lora_scale != 1.0:
                        module.scale_layer(lora_scale)

                    # For BC with previous PEFT versions, we need to check the signature
                    # of the `merge` method to see if it supports the `adapter_names` argument.
                    supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
                    if "adapter_names" in supported_merge_kwargs:
                        merge_kwargs["adapter_names"] = adapter_names
                    elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                        raise ValueError(
                            "The `adapter_names` argument is not supported with your PEFT version. "
                            "Please upgrade to the latest version of PEFT. `pip install -U peft`"
                        )

                    module.merge(**merge_kwargs)

        if fuse_image_encoder:
            if hasattr(self, "image_encoder"):
                fuse_image_encoder_lora(self.image_encoder, lora_scale, safe_fusing, adapter_names=adapter_names)
            if hasattr(self, "image_encoder2"):
                fuse_image_encoder_lora(self.image_encoder2, lora_scale, safe_fusing, adapter_names=adapter_names)

    def unfuse_lora(self, unfuse_unet: bool = True, unfuse_image_encoder: bool = True):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_image_encoder (`bool`, defaults to `True`):
                Whether to unfuse the image encoder LoRA parameters. If the image encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        if unfuse_unet:
            for module in unet.modules():
                if isinstance(module, BaseTunerLayer):
                    module.unmerge()

        def unfuse_image_encoder_lora(image_encoder):
            for module in image_encoder.modules():
                if isinstance(module, BaseTunerLayer):
                    module.unmerge()

        if unfuse_image_encoder:
            if hasattr(self, "image_encoder"):
                unfuse_image_encoder_lora(self.image_encoder)
            if hasattr(self, "image_encoder2"):
                unfuse_image_encoder_lora(self.image_encoder2)

        self.num_fused_loras -= 1

    def set_adapters_for_image_encoder(
        self,
        adapter_names: Union[List[str], str],
        image_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
        image_encoder_weights: Optional[Union[float, List[float], List[None]]] = None,
    ):
        """
        Sets the adapter layers for the image encoder.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            image_encoder (`torch.nn.Module`, *optional*):
                The image encoder module to set the adapter layers for. If `None`, it will try to get the `image_encoder`
                attribute.
            image_encoder_weights (`List[float]`, *optional*):
                The weights to use for the image encoder. If `None`, the weights are set to `1.0` for all the adapters.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        def process_weights(adapter_names, weights):
            # Expand weights into a list, one entry per adapter
            # e.g. for 2 adapters:  7 -> [7,7] ; [3, None] -> [3, None]
            if not isinstance(weights, list):
                weights = [weights] * len(adapter_names)

            if len(adapter_names) != len(weights):
                raise ValueError(
                    f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}"
                )

            # Set None values to default of 1.0
            # e.g. [7,7] -> [7,7] ; [3, None] -> [3,1]
            weights = [w if w is not None else 1.0 for w in weights]

            return weights

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        image_encoder_weights = process_weights(adapter_names, image_encoder_weights)
        image_encoder = image_encoder or getattr(self, "image_encoder", None)
        if image_encoder is None:
            raise ValueError(
                "The pipeline does not have a default `pipe.image_encoder` class. Please make sure to pass a `image_encoder` instead."
            )
        set_weights_and_activate_adapters(image_encoder, adapter_names, image_encoder_weights)

    def disable_lora_for_image_encoder(self, image_encoder: Optional["PreTrainedModel"] = None):
        """
        Disables the LoRA layers for the image encoder.

        Args:
            image_encoder (`torch.nn.Module`, *optional*):
                The image encoder module to disable the LoRA layers for. If `None`, it will try to get the
                `image_encoder` attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        image_encoder = image_encoder or getattr(self, "image_encoder", None)
        if image_encoder is None:
            raise ValueError("Image Encoder not found.")
        set_adapter_layers(image_encoder, enabled=False)

    def enable_lora_for_image_encoder(self, image_encoder: Optional["PreTrainedModel"] = None):
        """
        Enables the LoRA layers for the image encoder.

        Args:
            image_encoder (`torch.nn.Module`, *optional*):
                The image encoder module to enable the LoRA layers for. If `None`, it will try to get the `image_encoder`
                attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        image_encoder = image_encoder or getattr(self, "image_encoder", None)
        if image_encoder is None:
            raise ValueError("Image Encoder not found.")
        set_adapter_layers(self.image_encoder, enabled=True)

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[Union[float, Dict, List[float], List[Dict]]] = None,
    ):
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        adapter_weights = copy.deepcopy(adapter_weights)

        # Expand weights into a list, one entry per adapter
        if not isinstance(adapter_weights, list):
            adapter_weights = [adapter_weights] * len(adapter_names)

        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}"
            )

        # Decompose weights into weights for unet, image_encoder and image_encoder2
        unet_lora_weights, image_encoder_lora_weights, image_encoder2_lora_weights = [], [], []

        list_adapters = self.get_list_adapters()  # eg {"unet": ["adapter1", "adapter2"], "image_encoder": ["adapter2"]}
        all_adapters = {
            adapter for adapters in list_adapters.values() for adapter in adapters
        }  # eg ["adapter1", "adapter2"]
        invert_list_adapters = {
            adapter: [part for part, adapters in list_adapters.items() if adapter in adapters]
            for adapter in all_adapters
        }  # eg {"adapter1": ["unet"], "adapter2": ["unet", "image_encoder"]}

        for adapter_name, weights in zip(adapter_names, adapter_weights):
            if isinstance(weights, dict):
                unet_lora_weight = weights.pop("unet", None)
                image_encoder_lora_weight = weights.pop("image_encoder", None)
                image_encoder2_lora_weight = weights.pop("image_encoder2", None)

                if len(weights) > 0:
                    raise ValueError(
                        f"Got invalid key '{weights.keys()}' in lora weight dict for adapter {adapter_name}."
                    )

                if image_encoder2_lora_weight is not None and not hasattr(self, "image_encoder2"):
                    logger.warning(
                        "Lora weight dict contains image_encoder2 weights but will be ignored because pipeline does not have image_encoder2."
                    )

                # warn if adapter doesn't have parts specified by adapter_weights
                for part_weight, part_name in zip(
                    [unet_lora_weight, image_encoder_lora_weight, image_encoder2_lora_weight],
                    ["unet", "image_encoder", "image_encoder2"],
                ):
                    if part_weight is not None and part_name not in invert_list_adapters[adapter_name]:
                        logger.warning(
                            f"Lora weight dict for adapter '{adapter_name}' contains {part_name}, but this will be ignored because {adapter_name} does not contain weights for {part_name}. Valid parts for {adapter_name} are: {invert_list_adapters[adapter_name]}."
                        )

            else:
                unet_lora_weight = weights
                image_encoder_lora_weight = weights
                image_encoder2_lora_weight = weights

            unet_lora_weights.append(unet_lora_weight)
            image_encoder_lora_weights.append(image_encoder_lora_weight)
            image_encoder2_lora_weights.append(image_encoder2_lora_weight)

        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        # Handle the UNET
        unet.set_adapters(adapter_names, unet_lora_weights)

        # Handle the Image Encoder
        if hasattr(self, "image_encoder"):
            self.set_adapters_for_image_encoder(adapter_names, self.image_encoder, image_encoder_lora_weights)
        if hasattr(self, "image_encoder2"):
            self.set_adapters_for_image_encoder(adapter_names, self.image_encoder2, image_encoder2_lora_weights)

    def disable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Disable unet adapters
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet.disable_lora()

        # Disable image encoder adapters
        if hasattr(self, "image_encoder"):
            self.disable_lora_for_image_encoder(self.image_encoder)
        if hasattr(self, "image_encoder2"):
            self.disable_lora_for_image_encoder(self.image_encoder2)

    def enable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Enable unet adapters
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet.enable_lora()

        # Enable image encoder adapters
        if hasattr(self, "image_encoder"):
            self.enable_lora_for_image_encoder(self.image_encoder)
        if hasattr(self, "image_encoder2"):
            self.enable_lora_for_image_encoder(self.image_encoder2)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Args:
        Deletes the LoRA layers of `adapter_name` for the unet and image-encoder(s).
            adapter_names (`Union[List[str], str]`):
                The names of the adapter to delete. Can be a single string or a list of strings
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        unet.delete_adapters(adapter_names)

        for adapter_name in adapter_names:
            # Delete image encoder adapters
            if hasattr(self, "image_encoder"):
                delete_adapter_layers(self.image_encoder, adapter_name)
            if hasattr(self, "image_encoder2"):
                delete_adapter_layers(self.image_encoder2, adapter_name)

    def get_list_adapters(self) -> Dict[str, List[str]]:
        """
        Gets the current list of all available adapters in the pipeline.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        set_adapters = {}

        if hasattr(self, "image_encoder") and hasattr(self.image_encoder, "peft_config"):
            set_adapters["image_encoder"] = list(self.image_encoder.peft_config.keys())

        if hasattr(self, "image_encoder2") and hasattr(self.image_encoder2, "peft_config"):
            set_adapters["image_encoder2"] = list(self.image_encoder2.peft_config.keys())

        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        if hasattr(self, self.unet_name) and hasattr(unet, "peft_config"):
            set_adapters[self.unet_name] = list(self.unet.peft_config.keys())

        return set_adapters

    def set_lora_device(self, adapter_names: List[str], device: Union[torch.device, str, int]) -> None:
        """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[torch.device, str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        # Handle the UNET
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for unet_module in unet.modules():
            if isinstance(unet_module, BaseTunerLayer):
                for adapter_name in adapter_names:
                    unet_module.lora_A[adapter_name].to(device)
                    unet_module.lora_B[adapter_name].to(device)
                    # this is a param, not a module, so device placement is not in-place -> re-assign
                    if hasattr(unet_module, "lora_magnitude_vector") and unet_module.lora_magnitude_vector is not None:
                        unet_module.lora_magnitude_vector[adapter_name] = unet_module.lora_magnitude_vector[
                            adapter_name
                        ].to(device)

        # Handle the image encoder
        modules_to_process = []
        if hasattr(self, "image_encoder"):
            modules_to_process.append(self.image_encoder)

        if hasattr(self, "image_encoder2"):
            modules_to_process.append(self.image_encoder2)

        for image_encoder in modules_to_process:
            # loop over submodules
            for image_encoder_module in image_encoder.modules():
                if isinstance(image_encoder_module, BaseTunerLayer):
                    for adapter_name in adapter_names:
                        image_encoder_module.lora_A[adapter_name].to(device)
                        image_encoder_module.lora_B[adapter_name].to(device)
                        # this is a param, not a module, so device placement is not in-place -> re-assign
                        if (
                            hasattr(image_encoder_module, "lora_magnitude_vector")
                            and image_encoder_module.lora_magnitude_vector is not None
                        ):
                            image_encoder_module.lora_magnitude_vector[
                                adapter_name
                            ] = image_encoder_module.lora_magnitude_vector[adapter_name].to(device)



class StableDiffusionXLImageVariationClassEmbeddingMixin:
    r"""
    Load Class Embeddings Adapter for Image Variation.
    """

    def set_class_embedding_adapter(
        self,
        adapter: nn.Module,
        unet: Optional["UNet2DConditionModel"] = None,
        **kwargs,
    ):
        r"""
        Set class embedding adapter for image variation.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # 1. Set correct UNet model
        unet = unet or getattr(self, "unet", None)

        # 2. Check input
        if unet is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `self.unet` or passing a `UNet` of type `UNet2DConditionModel` for calling"
                f" `{self.load_class_embedding_adapter.__name__}`"
            )

        # 3. Set class embedding adapter
        assert unet.class_embedding is None, "Class embedding adapter is already set."
        unet.class_embedding = adapter

    def load_class_embedding_adapter(
        self,
        adapter_save_path: Union[str, os.PathLike, Dict],
        unet: Optional["UNet2DConditionModel"] = None,
        **kwargs,
    ):
        r"""
        Load class embedding adapter into the UNet.
        """
        # 1. Set correct UNet model
        unet = unet or getattr(self, "unet", None)

        # 2. Check input
        if unet is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `self.unet` or passing a `UNet` of type `UNet2DConditionModel` for calling"
                f" `{self.load_class_embedding_adapter.__name__}`"
            )

        # 3. Load state dict of class embedding adapter
        state_dict = torch.load(adapter_save_path)

        # 4. Load class embedding adapter
        unet.class_embedding.load_state_dict(state_dict)

    def unload_class_embedding_adapter(
        self,
        unet: Optional["UNet2DConditionModel"] = None,
        **kwargs,
    ):
        r"""
        Unload class embedding adapter from the UNet.
        """
        # 1. Set correct UNet model
        unet = unet or getattr(self, "unet", None)

        # 2. Check input
        if unet is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `self.unet` or passing a `UNet` of type `UNet2DConditionModel` for calling"
                f" `{self.unload_class_embedding_adapter.__name__}`"
            )

        # 3. Unload class embedding adapter
        unet.class_embedding = None


COCO_ID2LABEL = {
    "0": "N/A",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "N/A",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "N/A",
    "27": "backpack",
    "28": "umbrella",
    "29": "N/A",
    "30": "N/A",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "N/A",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "N/A",
    "67": "dining table",
    "68": "N/A",
    "69": "N/A",
    "70": "toilet",
    "71": "N/A",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "N/A",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush"
}

VOC_ID2LABEL = {
    "0": "aeroplane",
    "1": "bicycle",
    "2": "bird",
    "3": "boat",
    "4": "bottle",
    "5": "bus",
    "6": "car",
    "7": "cat",
    "8": "chair",
    "9": "cow",
    "10": "dining table",
    "11": "dog",
    "12": "horse",
    "13": "motorbike",
    "14": "person",
    "15": "potted plant",
    "16": "sheep",
    "17": "sofa",
    "18": "train",
    "19": "tv monitor"
}
