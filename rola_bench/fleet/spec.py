"""Shared experiment-spec loader for the config-driven benchmarks.

Every benchmark's experiments are declarative YAML under rola_bench/<bench>/experiments/<config>.yaml.
This loads one and flattens its structured sections to the box env vars the runners read, so the
spec file (committed) is the single source of truth — not shell env or code constants.

(MQAR additionally expands its spec to zoology TrainConfigs via rola_bench.mqar.build_configs.)
"""
import os

import yaml


def experiments_dir(bench):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), bench, "experiments")


def load(bench, config):
    path = config if str(config).endswith((".yaml", ".yml")) \
        else os.path.join(experiments_dir(bench), f"{config}.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def env_from(prefix, *sections):
    """Flatten spec sections to box env: key -> <PREFIX>_<KEY.upper()> (None skipped, all str)."""
    out = {}
    for sec in sections:
        for k, v in (sec or {}).items():
            if v is not None:
                out[f"{prefix}_{k.upper()}"] = str(v)
    return out
