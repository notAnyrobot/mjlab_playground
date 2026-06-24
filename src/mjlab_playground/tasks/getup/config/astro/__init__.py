from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import astro_getup_env_cfg
from .rl_cfg import astro_getup_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Getup-Flat-Astro",
    env_cfg=astro_getup_env_cfg(),
    play_env_cfg=astro_getup_env_cfg(play=True),
    rl_cfg=astro_getup_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
