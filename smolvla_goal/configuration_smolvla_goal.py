# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_goal")
@dataclass
class SmolVLAGoalConfig(SmolVLAConfig):
    """SmolVLA with target image conditioning.

    Adds N goal images to the prefix. Goal images go through the same vision
    tower as regular cameras, but their tokens have a learned `goal_type_embedding`
    added so the model can distinguish "current view" from "goal view".
    """

    # Number of goal images expected in the batch.
    # 0 = disabled (behaves exactly like SmolVLA).
    num_goal_images: int = 1

    # Key prefix used to identify goal images in the input_features dict.
    # Expected keys: "{goal_image_key_prefix}.0", "{goal_image_key_prefix}.1", ...
    goal_image_key_prefix: str = "observation.goal_image"