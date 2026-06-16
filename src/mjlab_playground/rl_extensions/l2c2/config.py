from dataclasses import dataclass


@dataclass
class L2C2Cfg:
    """L2C2 (Lipschitz-ratio) actor regularization (Kobayashi et al., 2022).

    Penalizes the ratio  ||mu(noisy) - mu(clean)||^2 / ||noisy - clean||^2
    so the actor's Lipschitz constant stays bounded.

    Reference: ``protomotions/agents/ppo/agent.py::calculate_extra_actor_loss``
    (ProtoMotions, lines 496-534).
    """

    enable: bool = False
    """Enable L2C2 regularization."""
    lambda_l2c2: float = 0.1
    """The coefficient for the L2C2 regularization."""
    clean_obs_suffix: str = "_clean"
    """Suffix appended to each actor observation group to find its clean counterpart."""
    clean_obs_groups: dict[str, str] | None = None
    """Resolved actor observation group to clean observation group mapping."""
