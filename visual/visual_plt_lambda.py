import argparse
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_timeline(timeline_path):
    steps, tasks, lambdas = [], [], []
    with open(timeline_path, "r") as timeline_file:
        for line in timeline_file:
            if not line.strip():
                continue
            step_str, task_str, lambda_str = line.strip().split("\t")
            steps.append(int(step_str))
            tasks.append(int(task_str))
            lambdas.append(float(lambda_str))
    return steps, tasks, lambdas


def find_task_boundaries(steps, tasks):
    boundaries = []
    if not steps:
        return boundaries
    last_task = tasks[0]
    last_step = steps[0]
    boundaries.append((last_task, last_step))
    for step, task in zip(steps, tasks):
        if task != last_task:
            boundaries.append((task, step))
            last_task = task
    return boundaries


def plot_lambda_timeline(timeline_path, output_path=None):
    steps, tasks, lambdas = load_timeline(timeline_path)
    if not steps:
        raise ValueError(f"No data loaded from {timeline_path}")

    unique_tasks = sorted(set(tasks))
    colors = cm.get_cmap("tab20", len(unique_tasks))
    color_map = {task_id: colors(idx) for idx, task_id in enumerate(unique_tasks)}

    fig, ax = plt.subplots(figsize=(12, 5))
    for task_id in unique_tasks:
        task_steps = [step for step, task in zip(steps, tasks) if task == task_id]
        task_lambdas = [lb for lb, task in zip(lambdas, tasks) if task == task_id]
        ax.plot(task_steps, task_lambdas, label=f"Task {task_id}", color=color_map[task_id], linewidth=1)

    for task_id, boundary_step in find_task_boundaries(steps, tasks):
        ax.axvline(boundary_step, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
        ax.text(boundary_step, max(lambdas), f"Task {task_id}", rotation=90, va="bottom", ha="right", fontsize=8)

    ax.set_xlabel("Global step")
    # Use standard mathtext with a single backslash so matplotlib can parse it
    ax.set_ylabel(r"$\lambda_t$")
    ax.set_title(r"Adaptive $\lambda_t$ across tasks")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    if output_path is None:
        base, _ = os.path.splitext(timeline_path)
        output_path = f"{base}.png"

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot lambda_t over training steps from timeline file.")
    parser.add_argument("timeline", help="Path to lambda_timeline-<param_stamp>.txt")
    parser.add_argument("--output", "-o", default=None, help="Output image path (png, pdf, etc.)")
    args = parser.parse_args()

    plot_lambda_timeline(args.timeline, args.output)
