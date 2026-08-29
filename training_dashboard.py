import re
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "training.log"
TRAINING_BATCH_SIZE = 16
TRAINING_SUBSET_SIZE = 12000
WAITING_AFTER_SECONDS = 15
GAUGE_MAX_FLOOR = 40.0

BATCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+),\s+Batch\s+(?P<batch>\d+)/(?P<batches>\d+),\s+Loss:\s+(?P<loss>[0-9.eE+-]+),\s+Time:\s+(?P<elapsed>[0-9.eE+-]+)s"
)
SUMMARY_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+):\s+train_loss=(?P<train_loss>[0-9.eE+-]+)\s+val_loss=(?P<val_loss>[0-9.eE+-]+)\s+val_auc=(?P<auc>[0-9.eE+-]+)\s+val_f1=(?P<f1>[0-9.eE+-]+)\s+time=(?P<time>[0-9.eE+-]+)s"
)

app = Flask(__name__)
state_lock = threading.Lock()
parser_state = {
    "offset": 0,
    "file_signature": None,
    "global_step": -1,
    "loss_history": deque(maxlen=3000),
    "epoch_history": {},
    "batch_times": deque(maxlen=100),
    "last_log_update": None,
    "last_line_at": None,
    "current": None,
    "validation_loss": None,
    "validation_auc": None,
    "validation_f1": None,
    "completed": False,
}


def _reset_parser():
    parser_state.update({
        "offset": 0,
        "file_signature": None,
        "global_step": -1,
        "loss_history": deque(maxlen=3000),
        "epoch_history": {},
        "batch_times": deque(maxlen=100),
        "last_log_update": None,
        "last_line_at": None,
        "current": None,
        "validation_loss": None,
        "validation_auc": None,
        "validation_f1": None,
        "completed": False,
    })


def _consume_line(line):
    batch_match = BATCH_PATTERN.search(line)
    if batch_match:
        values = batch_match.groupdict()
        epoch = int(values["epoch"])
        total_epochs = int(values["epochs"])
        batch = int(values["batch"])
        total_batches = int(values["batches"])
        loss = float(values["loss"])
        elapsed = float(values["elapsed"])
        global_step = (epoch - 1) * total_batches + batch
        previous = parser_state["current"]
        if previous and elapsed > previous["elapsed"] and global_step > previous["global_step"]:
            elapsed_per_batch = (elapsed - previous["elapsed"]) / (global_step - previous["global_step"])
            parser_state["batch_times"].append(elapsed_per_batch)
        parser_state["global_step"] = global_step
        parser_state["current"] = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "batch": batch,
            "total_batches": total_batches,
            "loss": loss,
            "elapsed": elapsed,
            "global_step": global_step,
            "timestamp": line[:23],
        }
        parser_state["loss_history"].append({"step": global_step, "loss": loss})
        parser_state["epoch_history"].setdefault(epoch, []).append({"step": global_step, "loss": loss})
        parser_state["last_log_update"] = time.time()
        parser_state["last_line_at"] = line[:23]

    summary_match = SUMMARY_PATTERN.search(line)
    if summary_match:
        values = summary_match.groupdict()
        parser_state["validation_loss"] = float(values["val_loss"])
        parser_state["validation_auc"] = float(values["auc"])
        parser_state["validation_f1"] = float(values["f1"])

    if "Training complete" in line or "TRAINING COMPLETE" in line:
        parser_state["completed"] = True


def _update_parser():
    if not LOG_PATH.exists():
        return
    signature = (LOG_PATH.stat().st_dev, LOG_PATH.stat().st_ino)
    if parser_state["file_signature"] != signature or LOG_PATH.stat().st_size < parser_state["offset"]:
        _reset_parser()
        parser_state["file_signature"] = signature

    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as log_file:
        log_file.seek(parser_state["offset"])
        for line in log_file:
            _consume_line(line)
        parser_state["offset"] = log_file.tell()


def _moving_average(values, window=20):
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def parse_training_log():
    with state_lock:
        _update_parser()
        current = parser_state["current"]
        if not current:
            return {"state": "UNKNOWN", "log_status": "Waiting for training.log", "loss_history": []}

        history = list(parser_state["loss_history"])
        losses = [point["loss"] for point in history]
        recent_times = list(parser_state["batch_times"])
        average_batch_time = sum(recent_times) / len(recent_times) if recent_times else None
        total_steps = current["total_epochs"] * current["total_batches"]
        completed_batches = current["global_step"] + 1
        overall_progress = min(1.0, completed_batches / total_steps) if total_steps else 0.0
        epoch_progress = current["batch"] / current["total_batches"] if current["total_batches"] else 0.0
        batches_per_second = 1 / average_batch_time if average_batch_time else 0.0
        images_per_second = batches_per_second * TRAINING_BATCH_SIZE
        throughput_samples = [TRAINING_BATCH_SIZE / batch_time for batch_time in recent_times if batch_time > 0]
        current_images_per_second = throughput_samples[-1] if throughput_samples else images_per_second
        observed_images_per_second = [
            TRAINING_BATCH_SIZE / batch_time for batch_time in recent_times if batch_time > 0
        ]
        eta = (total_steps - completed_batches) * average_batch_time if average_batch_time else None
        age = time.time() - (parser_state["last_log_update"] or time.time())
        state_name = "COMPLETED" if parser_state["completed"] else ("WAITING" if age > WAITING_AFTER_SECONDS else "TRAINING")
        smoothed = _moving_average(losses)
        epoch_history = {
            str(epoch): {"points": points, "average_loss": sum(point["loss"] for point in points) / len(points)}
            for epoch, points in parser_state["epoch_history"].items()
        }
        return {
            "state": state_name,
            "log_status": "Receiving updates" if age <= WAITING_AFTER_SECONDS else "Waiting for new log data",
            "epoch": current["epoch"],
            "total_epochs": current["total_epochs"],
            "batch": current["batch"],
            "total_batches": current["total_batches"],
            "epoch_progress": epoch_progress * 100,
            "overall_progress": overall_progress * 100,
            "loss": current["loss"],
            "moving_average_loss": smoothed[-1],
            "minimum_loss": min(losses),
            "average_loss": sum(losses) / len(losses),
            "change_from_previous": losses[-1] - losses[-2] if len(losses) > 1 else None,
            "batches_per_second": batches_per_second,
            "images_per_second": images_per_second,
            "current_images_per_second": current_images_per_second,
            "average_images_per_second": (
                sum(observed_images_per_second) / len(observed_images_per_second)
                if observed_images_per_second else images_per_second
            ),
            "peak_images_per_second": max(observed_images_per_second, default=images_per_second),
            "gauge_max": max(GAUGE_MAX_FLOOR, max(observed_images_per_second, default=0) * 1.25),
            "time_per_batch": average_batch_time,
            "elapsed_seconds": current["elapsed"],
            "eta_seconds": eta,
            "images_processed": completed_batches * TRAINING_BATCH_SIZE,
            "training_subset_size": TRAINING_SUBSET_SIZE,
            "dataset_progress": min(1.0, completed_batches * TRAINING_BATCH_SIZE / TRAINING_SUBSET_SIZE) * 100,
            "last_update": parser_state["last_line_at"],
            "loss_history": history,
            "smoothed_loss_history": [
                {"step": point["step"], "loss": value}
                for point, value in zip(history, smoothed)
            ],
            "epoch_history": epoch_history,
            "validation_loss": parser_state["validation_loss"],
            "validation_auc": parser_state["validation_auc"],
            "validation_f1": parser_state["validation_f1"],
            "batch_size": TRAINING_BATCH_SIZE,
        }


@app.get("/")
def dashboard():
    return render_template("training_dashboard.html")


@app.get("/api/status")
def status():
    return jsonify(parse_training_log())


def choose_port(preferred=8002):
    import socket

    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available localhost port found.")


if __name__ == "__main__":
    port = choose_port()
    print("=" * 48)
    print("LIVE TRAINING DASHBOARD")
    print("=" * 48)
    print("Open this URL in your browser:")
    print(f"http://127.0.0.1:{port}")
    print("Training process: NOT TOUCHED")
    print(f"Training log: {LOG_PATH}")
    print("=" * 48)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
