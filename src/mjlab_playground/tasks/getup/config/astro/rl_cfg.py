"""RL configuration for Dobot Astro getup task"""

from mjlab_playground.rl_extensions import (
    MjPgModelCfg,
    MjPgOnPolicyRunnerCfg,
    MjPgPpoAlgorithmCfg,
    RndCfg,
)
from mjlab_playground.rl_extensions.l2c2 import L2C2Cfg


def astro_getup_ppo_runner_cfg(
    *, l2c2: bool = False, rnd: bool = False
) -> MjPgOnPolicyRunnerCfg:
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
    l2c2_cfg = (
        L2C2Cfg(
            enable=True,
            lambda_l2c2=0.1,
            clean_obs_suffix="_clean",
        )
        if l2c2
        else None
    )
    rnd_cfg = (
        RndCfg(
            num_outputs=1,
            predictor_hidden_dims=(-1, -1),
            target_hidden_dims=(-1,),
            activation="elu",
            state_normalization=True,
            reward_normalization=False,
            weight=1.0,
            weight_schedule=None,
            learning_rate=0.001,
        )
        if rnd
        else None
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
        l2c2_cfg=l2c2_cfg,
        rnd_cfg=rnd_cfg,
    )

    return MjPgOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        experiment_name="astro_getup",
        wandb_project="mjlab_playground",
        save_interval=500,
        num_steps_per_env=32,  # 24
        max_iterations=50_000,
    )
