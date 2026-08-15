from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.core import load_config  # noqa: E402
from experiments.runner import run_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Stage B baseline with unified outputs.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/configs/stage_b_baselines.json")
    parser.add_argument(
        "--model",
        choices=["naive", "moving_average", "arima", "lstm", "minimalist_transformer"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config.resolve(), REPO_ROOT)
    seed = args.seed if args.seed is not None else int(config["seeds"][0])
    result = run_model(config, args.model, seed, REPO_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
