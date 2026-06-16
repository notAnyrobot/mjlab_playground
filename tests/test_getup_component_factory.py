"""Tests for getup observation component factories."""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab_playground.getup.component_factory import clean_actor_obs_groups
from mjlab_playground.getup.config.astro.env_cfgs import astro_getup_env_cfg


def _dummy_obs(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError


def _group(term_name: str, *, enable_corruption: bool = True) -> ObservationGroupCfg:
    return ObservationGroupCfg(
        terms={term_name: ObservationTermCfg(func=_dummy_obs)},
        concatenate_terms=False,
        concatenate_dim=0,
        enable_corruption=enable_corruption,
        history_length=3,
        flatten_history_dim=False,
        nan_policy="sanitize",
        nan_check_per_term=False,
    )


def test_clean_actor_obs_groups_builds_actor_clean_group() -> None:
    actor = _group("joint_pos")

    clean_groups = clean_actor_obs_groups({"actor": actor})

    assert tuple(clean_groups) == ("actor_clean",)
    actor_clean = clean_groups["actor_clean"]
    assert actor_clean.terms == actor.terms
    assert actor_clean.terms is not actor.terms
    assert actor_clean.concatenate_terms == actor.concatenate_terms
    assert actor_clean.concatenate_dim == actor.concatenate_dim
    assert actor_clean.enable_corruption is False
    assert actor_clean.history_length == actor.history_length
    assert actor_clean.flatten_history_dim == actor.flatten_history_dim
    assert actor_clean.nan_policy == actor.nan_policy
    assert actor_clean.nan_check_per_term == actor.nan_check_per_term


def test_clean_actor_obs_groups_builds_history_clean_group() -> None:
    actor = _group("joint_pos")
    actor_history = _group("joint_pos_history")

    clean_groups = clean_actor_obs_groups(
        {"actor": actor, "actor_history": actor_history}
    )

    assert tuple(clean_groups) == ("actor_clean", "actor_history_clean")
    assert clean_groups["actor_clean"].terms == actor.terms
    assert clean_groups["actor_history_clean"].terms == actor_history.terms
    assert clean_groups["actor_clean"].enable_corruption is False
    assert clean_groups["actor_history_clean"].enable_corruption is False


def test_astro_actor_clean_observations_match_actor_terms() -> None:
    cfg = astro_getup_env_cfg()

    assert cfg.observations["actor_clean"].terms == cfg.observations["actor"].terms
    assert cfg.observations["actor_clean"].terms is not cfg.observations["actor"].terms
    assert cfg.observations["actor_clean"].enable_corruption is False
