# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import math

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    pad_tensor,
)

from .configuration_smolvla_goal import SmolVLAGoalConfig


class VLAFlowMatchingGoal(VLAFlowMatching):
    """VLAFlowMatching with goal image conditioning.

    The last `num_goal_images` entries in the images list are treated as goals.
    Their patch tokens get a learned `goal_type_embedding` added before being
    concatenated into the prefix sequence.
    """

    def __init__(self, config: SmolVLAGoalConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        self.num_goal_images = config.num_goal_images

        if self.num_goal_images > 0:
            # Learned per-token bias added to every goal image token.
            # Shape (1, 1, D) broadcasts across (batch, num_patches, hidden_dim).
            # Small init so the model starts close to pretrained behavior.
            hidden_size = self.vlm_with_expert.config.text_config.hidden_size
            self.goal_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.goal_type_embedding, std=0.02)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as base embed_prefix, but adds `goal_type_embedding` to the last
        `num_goal_images` image entries.

        Assumes `prepare_images` places goal images at the end of the list.
        """
        embs = []
        pad_masks = []
        att_masks = []

        num_total = len(images)
        goal_start_idx = num_total - self.num_goal_images  # images at or after this index are goals

        for img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            is_goal = img_idx >= goal_start_idx and self.num_goal_images > 0

            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)

            # Normalize image embeddings (scale by sqrt(D))
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            # ** The one new line: add goal_type_embedding to goal image tokens **
            if is_goal:
                img_emb = img_emb + self.goal_type_embedding.to(dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        # --- Rest is identical to base embed_prefix ---
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks


class SmolVLAGoalPolicy(SmolVLAPolicy):
    """SmolVLA policy with target image conditioning.

    Swaps out the inner VLAFlowMatching for VLAFlowMatchingGoal and modifies
    prepare_images to ensure goal images land at the end of the list.
    """

    config_class = SmolVLAGoalConfig
    name = "smolvla_goal"

    def __init__(self, config: SmolVLAGoalConfig, **kwargs):
        # We want super().__init__ to do all its setup (rtc, reset, etc.) but
        # instantiate OUR model class. Easiest way: let super run, then replace.
        super().__init__(config, **kwargs)

        # Replace the VLAFlowMatching instance with our goal-aware version.
        # This is wasteful (we built the base model then throw it away) but
        # keeps our override surface minimal. The VLM weights inside the new
        # model are newly initialized here — from_pretrained() will load weights
        # into this final instance, so it works correctly for finetuning.
        self.model = VLAFlowMatchingGoal(config, rtc_processor=self.rtc_processor)

    def prepare_images(self, batch):
        """Same as base prepare_images but enforces that goal images come last.

        Regular camera keys are processed first (in dict order), goal keys
        (prefixed with config.goal_image_key_prefix) are processed after.
        """
        images = []
        img_masks = []

        goal_prefix = self.config.goal_image_key_prefix

        # Split keys into regular cameras and goal images, preserving config order within each group.
        camera_keys = [
            k for k in self.config.image_features
            if k in batch and not k.startswith(goal_prefix)
        ]
        goal_keys = sorted([
            k for k in self.config.image_features
            if k in batch and k.startswith(goal_prefix)
        ])
        missing_camera_keys = [
            k for k in self.config.image_features
            if k not in batch and not k.startswith(goal_prefix)
        ]

        if len(camera_keys) + len(goal_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Process regular cameras
        last_img = None
        last_mask = None
        for key in camera_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
            last_img, last_mask = img, mask

        # Empty-camera padding for missing regular cameras (same logic as base)
        for num_empty_cameras in range(len(missing_camera_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(last_img) * -1
            mask = torch.zeros_like(last_mask)
            images.append(img)
            img_masks.append(mask)

        # Process goal images LAST — VLAFlowMatchingGoal identifies them by position
        for key in goal_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks