#!/usr/bin/env python3
"""Compare training time and GPU memory usage between baseline and DGR runs.

In this project, "DGR" refers to the adaptive regularization strength plugin
implemented in the continual learner (not generative replay). This script runs
the baseline regularization methods (EWC / SI / MAS / RWALK) with and without
the plugin enabled and summarizes their runtime and GPU memory usage.
"""
import copy
import os
import numpy as np

import main
from params import options
from params.param_values import check_for_errors, set_default_values, set_method_options
from params.param_stamp import get_param_stamp_from_args


METHOD_CHOICES = ('ewc', 'si', 'mas', 'rwalk')


def handle_inputs():
    kwargs = {'comparison': True}
    parser = options.define_args(
        filename="compare_resources",
        description='Compare runtime and memory for baseline and DGR (adaptive regularization).',
    )
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)

    parser.add_argument('--skip-baseline', action='store_true', help='skip run without adaptive regularization')
    parser.add_argument('--skip-dgr', action='store_true', help='skip run with adaptive regularization enabled')
    parser.add_argument('--summary-tag', type=str, default=None, help='extra tag for the summary filename')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=list(METHOD_CHOICES),
        choices=METHOD_CHOICES,
        help='which baseline regularization methods to compare (default: all)'
    )

    # Adaptive regularization hyperparameters (mirrored from main.py)
    parser.add_argument("--lambda_0", type=float, default=0.3, help="Initial lambda value")
    parser.add_argument('--scaling_power', type=float, default=0.33,
                        help='Scaling power for adaptive regularization (default: 0.33)')
    parser.add_argument('--ablation_fixed_numerator', action='store_true', default=False,
                        help='Fix numerator to constant value for ablation study')
    parser.add_argument('--fixed_numerator_value', type=float, default=0.1,
                        help='Fixed numerator value (default: 0.1)')
    parser.add_argument('--ablation_fixed_denominator', action='store_true', default=False,
                        help='Fix denominator to constant value for ablation study')
    parser.add_argument('--fixed_denominator_value', type=float, default=10.0,
                        help='Fixed denominator value (default: 10.0)')

    args = parser.parse_args()
    # In comparison mode the replay option isn't exposed; default to no replay so param-stamps work.
    if not hasattr(args, 'replay'):
        args.replay = 'none'
    # Ensure regularization strength is present so downstream helpers (param-stamp) can use defaults.
    if not hasattr(args, 'reg_strength'):
        args.reg_strength = None
    # For resource comparison we always train new models (unless logs already exist).
    if not hasattr(args, 'train'):
        args.train = True
    args.log_per_context = True
    set_default_values(args, also_hyper_params=True)
    check_for_errors(args, **kwargs)
    return args


def _read_scalar(path):
    """Read a float from a text file if it exists."""
    if os.path.isfile(path):
        with open(path, 'r') as f:
            content = f.readline().strip()
        try:
            return float(content)
        except ValueError:
            return None
    return None


def _prepare_method_args(base_args, method_name, use_adaptive):
    """Reset method flags, enable the requested method and adaptive toggle, then re-apply defaults."""
    run_args = copy.deepcopy(base_args)

    # clear method flags so only the requested baseline is active
    for flag in METHOD_CHOICES:
        if hasattr(run_args, flag):
            setattr(run_args, flag, False)
    if hasattr(run_args, 'importance_weighting'):
        run_args.importance_weighting = None
    if hasattr(run_args, 'weight_penalty'):
        run_args.weight_penalty = False

    # enable the requested baseline and adaptive toggle
    setattr(run_args, method_name, True)
    run_args.use_adaptive = use_adaptive

    # recompute derived defaults for this configuration
    set_method_options(run_args)
    set_default_values(run_args, also_hyper_params=True)
    check_for_errors(run_args, comparison=True)
    return run_args


def run_variant(base_args, method_name, use_adaptive, label):
    """Run a single variant (baseline or DGR) for all seeds."""
    seed_list = list(range(base_args.seed, base_args.seed + base_args.n_seeds))
    records = []

    for seed in seed_list:
        run_args = _prepare_method_args(base_args, method_name, use_adaptive)
        run_args.seed = seed
        run_args.distill = False
        run_args.time = True
        run_args.track_resources = True

        print(f"\n--- Running {label} (seed {seed}) ---")
        param_stamp = get_param_stamp_from_args(run_args)
        time_path = os.path.join(run_args.r_dir, f"time-{param_stamp}.txt")
        mem_path = os.path.join(run_args.r_dir, f"memory-{param_stamp}.txt")

        has_time = os.path.isfile(time_path)
        has_mem = os.path.isfile(mem_path)
        if has_time and has_mem:
            print(f"Found existing resource logs for seed {seed}, skipping training.")
        else:
            main.run(run_args)

        records.append(
            {
                'seed': seed,
                'param_stamp': param_stamp,
                'time': _read_scalar(time_path),
                'memory': _read_scalar(mem_path),
            }
        )

    return records


def summarize_records(label, records):
    """Convert raw records into readable summary lines."""
    lines = [label]
    times = [rec['time'] for rec in records if rec['time'] is not None]
    mems = [rec['memory'] for rec in records if rec['memory'] is not None]

    for rec in records:
        time_txt = 'N/A' if rec['time'] is None else f"{rec['time']:.1f}s"
        mem_txt = 'N/A' if rec['memory'] is None else f"{rec['memory'] / (1024 ** 2):.1f} MB"
        lines.append(f"  seed {rec['seed']}: time {time_txt}, peak GPU memory {mem_txt}")

    if times:
        lines.append(f"  mean time: {np.mean(times):.1f}s")
    if mems:
        lines.append(f"  mean peak GPU memory: {np.mean(mems) / (1024 ** 2):.1f} MB")

    return lines


def main_entry():
    args = handle_inputs()

    summary_lines = []
    base_name = args.summary_tag if args.summary_tag is not None else (
        f"{args.experiment}-{args.contexts}{args.scenario}-baseline-vs-dgr"
    )
    summary_path = os.path.join(args.r_dir, f"resources-{base_name}.txt")

    for method in args.methods:
        header = method.upper()

        if not args.skip_baseline:
            baseline_records = run_variant(
                args, method_name=method, use_adaptive=False,
                label=f"{header} baseline (no adaptive regularization)",
            )
            summary_lines.extend(summarize_records(f"{header} baseline (use_adaptive=False):", baseline_records))
            summary_lines.append("")

        if not args.skip_dgr:
            dgr_records = run_variant(
                args, method_name=method, use_adaptive=True,
                label=f"{header} + DGR (adaptive regularization)",
            )
            summary_lines.extend(summarize_records(f"{header} + DGR (use_adaptive=True):", dgr_records))
            summary_lines.append("")

    if summary_lines:
        os.makedirs(args.r_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        print('\n'.join(summary_lines))
        print(f"\nSaved resource summary to {summary_path}")
    else:
        print('No runs executed. Use --skip-baseline/--skip-dgr to control which runs to launch.')


if __name__ == '__main__':
    main_entry()