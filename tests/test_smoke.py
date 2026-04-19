"""Smoke test: verifies SmolVLAGoalPolicy instantiates correctly."""

import torch

from smolvla_goal import SmolVLAGoalConfig, SmolVLAGoalPolicy


def test_config_instantiates():
    config = SmolVLAGoalConfig(num_goal_images=1)
    assert config.num_goal_images == 1
    assert config.goal_image_key_prefix == "observation.goal_image"
    # Inherited from SmolVLAConfig
    assert config.chunk_size == 50
    print("✓ SmolVLAGoalConfig instantiates with correct defaults")


def test_goal_embedding_shape():
    """Build the policy (no weights loaded) and check goal_type_embedding exists."""
    from lerobot.configs import FeatureType, PolicyFeature

    config = SmolVLAGoalConfig(
        num_goal_images=1,
        load_vlm_weights=False,  # don't download VLM weights for a smoke test
    )
    # Minimal valid input_features: one camera, one goal image, state, action
    config.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.goal_image.0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    policy = SmolVLAGoalPolicy(config)

    # Verify the goal embedding exists and has the right shape
    assert hasattr(policy.model, "goal_type_embedding"), "goal_type_embedding missing"
    hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
    expected_shape = (1, 1, hidden)
    assert policy.model.goal_type_embedding.shape == expected_shape, (
        f"Expected {expected_shape}, got {policy.model.goal_type_embedding.shape}"
    )
    # Verify it's a trainable parameter
    assert policy.model.goal_type_embedding.requires_grad, "goal_type_embedding should be trainable"

    print(f"✓ goal_type_embedding shape: {policy.model.goal_type_embedding.shape}")
    print(f"✓ goal_type_embedding requires_grad: {policy.model.goal_type_embedding.requires_grad}")
    print(f"✓ goal_type_embedding initial mean: {policy.model.goal_type_embedding.mean().item():.6f}")


if __name__ == "__main__":
    test_config_instantiates()
    test_goal_embedding_shape()
    print("\nAll smoke tests passed.")
