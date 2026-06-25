"""RL configuration for Dobot Astro getup task"""

from mjlab_playground.rl_extensions import (
    MjPgModelCfg,
    MjPgOnPolicyRunnerCfg,
    MjPgPpoAlgorithmCfg,
)
from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg


def astro_getup_ppo_runner_cfg() -> MjPgOnPolicyRunnerCfg:
    """Create RL runner configuration for Astro getup task."""

    actor: MjPgModelCfg = MjPgModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    )
    critic: MjPgModelCfg = MjPgModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    l2c2: L2C2Cfg = L2C2Cfg(
        enable=True,
        lambda_l2c2=0.1,
        clean_obs_suffix="_clean",
    )
    algorithm: MjPgPpoAlgorithmCfg = MjPgPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        l2c2_cfg=l2c2,
    )

    return MjPgOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        experiment_name="astro_getup",
        wandb_project="mjlab_playground",
        save_interval=500,
        num_steps_per_env=32,   # 24
        max_iterations=50_000,
    )
