"""训练和评估 LSTM-SAC"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / 'configs/lstm_sac_procs.yaml'
SAC_CONFIG_PATH = ROOT / 'configs/mlp_sac_procs.yaml'

if __name__ == '__main__':
    for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_key, '1')

from env import close_envs, make_envs
from env.procs_env import parse_history_length
from utils.config import load_config
from utils.run_utils import (
    add_robustness_arguments,
    assert_spaces,
    configure_actuator,
    configure_irregularity,
    configure_plant,
    configure_speed_run,
    eval_condition_name,
    eval_run_name,
    evaluate_lstm_sac,
    finite_action_check,
    load_path,
    print_device,
    train_sac_agent,
    validate_device,
    validate_lstm_sac_test_model,
    validate_sac_training_steps,
)


def require_history_length(config) -> int:
    env_kwargs = getattr(config, 'env_kwargs', None)
    if 'history_length' not in (env_kwargs or {}):
        raise ValueError('LSTM-SAC requires env_kwargs.history_length to be set to a positive integer')
    history_length = parse_history_length(env_kwargs['history_length'], default=0)
    if history_length <= 0:
        raise ValueError(f'LSTM-SAC requires a positive history_length, got {env_kwargs["history_length"]!r}')
    return history_length


def observation_shape(history_length: int) -> tuple[int, ...]:
    if history_length <= 0:
        raise ValueError(f'history_length must be positive, got {history_length}')
    return (history_length, 5)


def run(
    command: str,
    steps: int | None,
    model: str | None,
    sac_model: str | None,
    speed: float | None = None,
    seed: int | None = None,
    eval_seed: int = 101,
    irregularity: bool | None = None,
    wear_amplitude: float | None = None,
    wear_wavelength: float | None = None,
    actuator_delta: float | None = None,
    actuator_noise_std: float | None = None,
    pantograph: int | None = None,
    catenary: int | None = None,
) -> None:
    validate_lstm_sac_test_model(command, model)
    model_load = load_path(model) if command == 'test' else None
    sac_model_load = load_path(sac_model) if command == 'test' and sac_model else None

    overrides = {'test_mode': command == 'test'}
    if steps is not None:
        overrides['running_steps'] = steps
    if seed is not None:
        overrides.update(seed=seed, env_seed=seed)
    if command == 'test':
        overrides['parallels'] = 1
        overrides['vectorize'] = 'DummyVecEnv'

    config = load_config(CONFIG_PATH, overrides)
    raw_env_kwargs = getattr(config, 'env_kwargs', None)
    if raw_env_kwargs is not None and not isinstance(raw_env_kwargs, dict):
        raise TypeError(f'env_kwargs must be None or a dictionary, got {type(raw_env_kwargs).__name__}')
    history_length = require_history_length(config)
    env_kwargs, speed_dir = configure_speed_run(config, speed)
    env_kwargs = configure_irregularity(
        env_kwargs,
        irregularity=irregularity,
        wear_amplitude=wear_amplitude,
        wear_wavelength=wear_wavelength,
    )
    env_kwargs = configure_actuator(
        env_kwargs,
        actuator_delta=actuator_delta,
        actuator_noise_std=actuator_noise_std,
    )
    env_kwargs = configure_plant(env_kwargs, pantograph=pantograph, catenary=catenary)
    print(f'condition: {eval_condition_name(env_kwargs)}')
    config.env_kwargs = env_kwargs
    validate_sac_training_steps(
        command,
        config.running_steps,
        config.parallels,
        start_training=config.start_training,
        training_frequency=config.training_frequency,
    )
    validate_device(config.device)

    envs = make_envs(config)
    agent = None
    sac_agent = None
    try:
        assert_spaces(envs, obs_shape=observation_shape(history_length))
        from algorithm.SAC import SACAgent

        agent = SACAgent(config, envs)
        print_device(agent)

        if not config.test_mode:
            train_sac_agent(
                agent,
                running_steps=config.running_steps,
                num_envs=envs.num_workers,
                eval_interval=config.eval_interval,
                env_kwargs=env_kwargs,
                history_length=history_length,
            )
        else:
            load_dir, load_name = model_load
            agent.load_model(load_dir, load_name)
            finite_action_check(agent)

            if sac_model_load is not None:
                sac_config = load_config(
                    SAC_CONFIG_PATH,
                    {'test_mode': True, 'parallels': 1, 'vectorize': 'DummyVecEnv'},
                )
                sac_config.device = config.device
                sac_history_length = require_history_length(sac_config)
                sac_envs = make_envs(sac_config)
                try:
                    assert_spaces(sac_envs, obs_shape=(sac_history_length, 5))
                    sac_agent = SACAgent(sac_config, sac_envs)
                    sac_load_dir, sac_load_name = sac_model_load
                    sac_agent.load_model(sac_load_dir, sac_load_name)
                finally:
                    close_envs(sac_envs)
            else:
                sac_history_length = 0

            evaluate_lstm_sac(
                agent,
                sac_agent,
                env_kwargs,
                ROOT
                / 'results'
                / 'eval'
                / 'lstm_sac'
                / speed_dir
                / eval_run_name(load_dir)
                / eval_condition_name(env_kwargs)
                / f'seed_{eval_seed}',
                history_length=history_length,
                sac_history_length=sac_history_length if sac_agent is not None else 0,
                seed=eval_seed,
            )
    finally:
        if sac_agent is not None:
            sac_agent.finish()
        if agent is not None:
            agent.finish()
        close_envs(envs)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description='LSTM-SAC train/test on PROCS')
    parser.add_argument('command', choices=('train', 'test'))
    parser.add_argument('--steps', type=int, default=None, help='override running_steps')
    parser.add_argument('--model', type=str, default=None, help='LSTM-SAC checkpoint file or seed_* directory')
    parser.add_argument(
        '--speed',
        type=float,
        nargs='+',
        default=None,
        help='fixed speed in km/h; each speed is trained and saved as its own model',
    )
    parser.add_argument('--seed', type=int, default=None, help='override algorithm and environment seed')
    parser.add_argument('--eval-seed', type=int, default=101, help='test rollout seed for stochastic environments')
    add_robustness_arguments(parser)
    parser.add_argument(
        '--sac-model',
        type=str,
        default=None,
        help='optional baseline SAC checkpoint for three-way comparison',
    )
    args = parser.parse_args()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    speeds = args.speed or [None]
    if args.command != 'train' and len(speeds) > 1:
        parser.error('test accepts a single --speed')
    for speed in speeds:
        if speed is not None:
            print(f'{args.command} {speed:g} km/h')
        run(
            args.command,
            args.steps,
            args.model,
            args.sac_model,
            speed,
            args.seed,
            args.eval_seed,
            args.irregularity,
            args.wear_amplitude,
            args.wear_wavelength,
            args.actuator_delta,
            args.actuator_noise_std,
            args.pantograph,
            args.catenary,
        )


if __name__ == '__main__':
    main()
