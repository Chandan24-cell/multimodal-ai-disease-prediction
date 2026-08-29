import argparse
import os
import re
import time
from collections import deque
from datetime import datetime

BATCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+),\s+Batch\s+(?P<batch>\d+)/(?P<batches>\d+),\s+Loss:\s+(?P<loss>[0-9.eE+-]+),\s+Time:\s+(?P<elapsed>[0-9.eE+-]+)s"
)
SUMMARY_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+):\s+train_loss=(?P<train_loss>[0-9.eE+-]+)\s+val_loss=(?P<val_loss>[0-9.eE+-]+)\s+val_auc=(?P<auc>[0-9.eE+-]+)\s+val_f1=(?P<f1>[0-9.eE+-]+)\s+time=(?P<time>[0-9.eE+-]+)s"
)


def parse_log(log_path, state):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(state["offset"])
            for line in log_file:
                batch_match = BATCH_PATTERN.search(line)
                if batch_match:
                    values = batch_match.groupdict()
                    state.update(
                        epoch=int(values["epoch"]),
                        epochs=int(values["epochs"]),
                        batch=int(values["batch"]),
                        batches=int(values["batches"]),
                        loss=float(values["loss"]),
                        elapsed=float(values["elapsed"]),
                    )
                    state["losses"].append((state["global_step"], state["loss"]))
                    state["global_step"] += 1

                summary_match = SUMMARY_PATTERN.search(line)
                if summary_match:
                    values = summary_match.groupdict()
                    state["val_loss"] = float(values["val_loss"])
                    state["val_auc"] = float(values["auc"])
                    state["val_f1"] = float(values["f1"])

            state["offset"] = log_file.tell()
    except FileNotFoundError:
        state["missing"] = True


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def render_graph(losses, width=58, height=8):
    if not losses:
        return ["  waiting for loss records..."]

    points = list(losses)[-width:]
    values = [loss for _, loss in points]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        maximum += 1.0

    rows = []
    for row in range(height):
        threshold = maximum - (maximum - minimum) * row / (height - 1)
        line = []
        for value in values:
            line.append("●" if abs(value - threshold) <= (maximum - minimum) / height else " ")
        rows.append(f" {threshold:6.3f} ┤{''.join(line)}")
    rows.append(f" {minimum:6.3f} └{'─' * len(values)}")
    return rows


def render(state, log_path):
    epoch = state["epoch"]
    batch = state["batch"]
    batches = state["batches"]
    epochs = state["epochs"]
    completed_batches = max(0, (epoch - 1) * batches + batch + 1)
    total_batches = max(1, epochs * batches)
    progress = min(1.0, completed_batches / total_batches)
    speed = completed_batches / state["elapsed"] if state["elapsed"] > 0 else 0.0
    remaining = max(0, total_batches - completed_batches)
    eta = remaining / speed if speed > 0 else None
    bar_width = 38
    filled = int(progress * bar_width)

    lines = [
        "\033[2J\033[H",
        "╔══════════════════════════════════════════════════════════════╗",
        "║              MULTIMODAL AI TRAINING LIVE                    ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║ Epoch       : {epoch:>3} / {epochs:<3}     Batch: {batch:>5} / {batches:<5}       ║",
        f"║ Progress    : [{'█' * filled}{'░' * (bar_width - filled)}] {progress * 100:6.2f}% ║",
        f"║ Loss        : {state['loss']:>10.5f}     Speed: {speed:>7.3f} batches/sec ║",
        f"║ Elapsed     : {format_duration(state['elapsed']):>10}     ETA: {format_duration(eta):>10}       ║",
        f"║ Log         : {log_path[:48]:<48} ║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║ LIVE LOSS HISTORY                                             ║",
    ]
    lines.extend(f"║{graph:<62}║" for graph in render_graph(state["losses"]))
    lines.extend([
        "╠══════════════════════════════════════════════════════════════╣",
        f"║ Validation  : loss={state['val_loss']!s:<10} AUC={state['val_auc']!s:<10} F1={state['val_f1']!s:<10} ║",
        "║ Press Ctrl-C to stop this monitor. Training is independent.  ║",
        "╚══════════════════════════════════════════════════════════════╝",
    ])
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Read-only live monitor for training.log")
    parser.add_argument("--log-file", default="training.log")
    parser.add_argument("--interval", type=float, default=1.5)
    args = parser.parse_args()

    state = {
        "offset": 0,
        "missing": False,
        "epoch": 0,
        "epochs": 0,
        "batch": 0,
        "batches": 0,
        "loss": 0.0,
        "elapsed": 0.0,
        "global_step": 0,
        "val_loss": "--",
        "val_auc": "--",
        "val_f1": "--",
        "losses": deque(maxlen=500),
    }

    try:
        while True:
            parse_log(args.log_file, state)
            if state["missing"]:
                print(f"Waiting for training log: {args.log_file}", flush=True)
            else:
                render(state, args.log_file)
            time.sleep(max(0.25, args.interval))
    except KeyboardInterrupt:
        print("\nTraining monitor stopped. The training process was not touched.")


if __name__ == "__main__":
    main()
