"""Create and run W&B hyperparameter sweeps."""

import argparse
from pathlib import Path

import wandb
import yaml

from src.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / 'config' / 'sweep_config.yaml'


def train_sweep():
    run = wandb.init()
    try:
        config = dict(run.config)

        print(f"\n{'=' * 70}")
        print("Running configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 70)

        train_model(
            config,
            project_name=run.project,
            use_wandb=True,
            wandb_run=run
        )
    finally:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='Run W&B Sweep')
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help='Path to sweep config file'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help='Number of runs for this agent'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='dl_assignment',
        help='W&B project name'
    )
    parser.add_argument(
        '--create_only',
        action='store_true',
        help="Only create sweep; do not run an agent"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open('r', encoding='utf-8') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project
    )

    print(f"\n{'=' * 70}")
    print(f"Created sweep with ID: {sweep_id}")
    print("=" * 70)
    print("\nTo run this sweep:")
    print(f"  wandb agent {sweep_id}")
    print("\nOr run multiple agents in parallel from separate terminals.")
    print(f"\n{'=' * 70}\n")

    if not args.create_only:
        print(f"Starting agent with {args.count} runs...")
        wandb.agent(sweep_id, function=train_sweep, count=args.count)
        print("\nSweep agent completed.")


if __name__ == "__main__":
    main()
