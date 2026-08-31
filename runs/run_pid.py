"""Evaluate traditional PID control on PROCS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.procs_core import WEAR_AMPLITUDE, WEAR_WAVELENGTH
from utils.run_utils import eval_condition_name, evaluate_pid


def main() -> None:
    parser = argparse.ArgumentParser(description='PID evaluation on PROCS')
    parser.add_argument('--speed', type=float, default=None, help='fixed speed in km/h; omit to evaluate 200/250/300')
    parser.add_argument('--kp', type=float, default=0.1)
    parser.add_argument('--ki', type=float, default=0.05)
    parser.add_argument('--kd', type=float, default=0.001)
    parser.add_argument('--action-max', type=float, default=50.0)
    parser.add_argument('--control-freq', type=float, default=100.0,
                        help='与 SAC 配置保持一致（100 Hz）才是同频基线；如需 500 Hz 显式传入')
    parser.add_argument('--eval-seed', type=int, default=101)
    parser.add_argument('--irregularity', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--WEAR_AMPLITUDE', '--wear-amplitude', dest='wear_amplitude', type=float, default=WEAR_AMPLITUDE
    )
    parser.add_argument(
        '--WEAR_WAVELENGTH', '--wear-wavelength', dest='wear_wavelength', type=float, default=WEAR_WAVELENGTH
    )
    args = parser.parse_args()

    speed_dir = f'{args.speed:g}kmh' if args.speed is not None else '200-250kmh'
    env_kwargs = {
        'speed_kmh': args.speed if args.speed is not None else 250.0,
        'speed_range_kmh': None if args.speed is not None else (200.0, 250.0),
        'control_freq': args.control_freq,
        'action_max': args.action_max,
        'action_rate_weight': 0.1,
        'wear_amplitude': args.wear_amplitude,
        'wear_wavelength': args.wear_wavelength,
        'irregularity': args.irregularity,
    }
    evaluate_pid(
        env_kwargs,
        ROOT
        / 'results'
        / 'eval'
        / 'pid'
        / speed_dir
        / eval_condition_name(env_kwargs)
        / f'seed_{args.eval_seed}',
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        seed=args.eval_seed,
    )


if __name__ == '__main__':
    main()
