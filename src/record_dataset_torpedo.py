import os
import yaml
import holoocean
import time
import multiprocessing as mp
import traceback
import threading
import queue
import sys

from lib.scenario_builder import ScenarioConfig
from lib.worlds import World
from lib.rover import Rover

from telemetry.parsing import parse_pose
from telemetry.estimation import (
    parse_velocity,
    parse_depth,
    estimate_motion_state,
)

from utils.convert import pose_to_csv_fields, velocity_to_csv_fields
from utils.writer import DatasetWriter
from utils.trajectories import (
    ForwardTrajectory,
    LateralTrajectory,
    LateralOppositeTrajectory,
    ZigZagTrajectory,
)

# ===================== CONFIG =====================

DATASET_ROOT = "dataset/runs"

OBJECT_CLASS = "torpedo" 

MAP_NAME = "World.Custom"

START_POSITIONS = [  
    (-13.0, 45.0, -30.0),
    (-14.0, 45.0, -30.0),
    (-12.0, 45.0, -30.0),
]

DEPTHS_Z = [-30.0, -25.0, -20.0, -15.0]
STOP_Y = -19.16
DESCENT_PER_FRAME = 0

FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"
SONAR_KEY = "ImagingSonar"

TRAJECTORIES = [
    ForwardTrajectory(),
    LateralTrajectory(),
    LateralOppositeTrajectory(),
    ZigZagTrajectory(),
]

SENSOR_MAP = {
    "Pose": "PoseSensor",
    "Velocity": "VelocitySensor",
    "IMU": "IMUSensor",
    "Depth": "RangeFinderSensor",
}

# ===================== CORE =====================

def rotation_for_trajectory(traj):
    if traj.name == "lateral":
        return [0, 0, -360]
    elif traj.name == "lateral_opposite":
        return [0, 0, -180]
    else:
        return [0, 0, -90]

def run_single(start_pos, start_z, traj, run_idx):

    run_id = f"run_{run_idx:04d}"
    x0, y0, _ = start_pos

    print(f"\n{run_id} | start=({x0},{y0},{start_z}) | traj={traj.name}")

    run_metadata = {
        "run_id": run_id,
        "dataset_version": "v1.0",
        "map": MAP_NAME,
        "primary_object": OBJECT_CLASS,

        "initial_position": {
            "x": x0,
            "y": y0,
            "z": start_z,
        },
        "initial_depth_m": abs(start_z),
        "motion_pattern": traj.name,
        "control_mode": "thruster",

        "vertical_motion": {
            "enabled": False,
            "method": "none",
            "vertical_thrust": 0.0,
        },

        "termination": {
            "type": "y_threshold",
            "max_frames": None,
            "y_threshold": STOP_Y,
        },

        "sensors": {
            "front_camera": FRONT_CAM,
            "bottom_camera": BOTTOM_CAM,
            "sonar": SONAR_KEY,
            "altitude_sensor": "RangeFinderSensor",
        },

        "environment": {
            "water_fog": {
                "enabled": True,
                "density": 5.0,
                "distance": 5.0,
            }
        },

        "notes": "CustomTorpedoes run (torpedo detection)",  
    }

    run_path = os.path.join(DATASET_ROOT, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    rov = Rover.BlueROV2(
        name="rov0",
        location=[x0, y0, start_z],
        rotation=rotation_for_trajectory(traj),
        control_scheme=0,
    )

    scenario = (
        ScenarioConfig("DatasetRun")
        .set_world(World.Example)
        .add_agent(rov)
    )

    writer = DatasetWriter(
        root=DATASET_ROOT,
        run_id=run_id,
        front_cam_key=FRONT_CAM,
        bottom_cam_key=BOTTOM_CAM,
        sonar_key=SONAR_KEY,
        pose_to_csv_fields=pose_to_csv_fields,
        velocity_to_csv_fields=velocity_to_csv_fields,
    )

    with holoocean.make(
        scenario_cfg=scenario.to_dict(),
        show_viewport=False,
        ticks_per_sec=30,
        frames_per_sec=True,
        start_world=False,
    ) as env:

        env.tick(2)
        last = {}
        t = 0

        while True:

            cmd = traj.command(t)
            t += 1

            state = env.step(cmd)

            for k, s in SENSOR_MAP.items():
                if s in state:
                    last[k] = state[s]

            pose = parse_pose(last.get("Pose"))
            if pose is None:
                continue

            _, y, _ = pose["pos"]

            if y < STOP_Y:
                print(f" Stop condition reached (y={y:.2f})")
                break

            if SONAR_KEY not in state:
                continue

            telemetry = {
                "pose": pose,
                "velocity": parse_velocity(last.get("Velocity")),
                "altitude": parse_depth(last.get("Depth")),
                "motion": estimate_motion_state(last.get("IMU")),
            }

            writer.write_frame(state, telemetry)

    writer.close()
    print(f"{run_id} complete")


RUN_TIMEOUT_SEC = 800
def _worker_run_single(args, q):
    start_pos, start_z, traj, run_idx = args
    try:
        run_single(start_pos, start_z, traj, run_idx)
        q.put(("ok", None))
    except Exception:
        q.put(("err", traceback.format_exc()))

def _stdin_listener(out_q: "queue.Queue[str]", stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            line = sys.stdin.readline()
            if not line: 
                break
            out_q.put(line.strip().lower())
        except Exception:
            break


def run_single_with_timeout_or_manual_kill(start_pos, start_z, traj, run_idx, timeout_sec=RUN_TIMEOUT_SEC):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_run_single, args=((start_pos, start_z, traj, run_idx), q))
    p.start()

    stop_evt = threading.Event()
    kb_q: "queue.Queue[str]" = queue.Queue()
    th = threading.Thread(target=_stdin_listener, args=(kb_q, stop_evt), daemon=True)
    th.start()

    t0 = time.time()

    try:
        while True:
            if not p.is_alive():
                break

            if (time.time() - t0) >= timeout_sec:
                p.terminate()
                p.join(10)
                return ("timeout", f"Run {run_idx} killed after {timeout_sec}s")

            try:
                cmd = kb_q.get_nowait()
                if cmd in ("q", "quit", "stop", "kill"):
                    p.terminate()
                    p.join(10)
                    return ("timeout", f"Run {run_idx} manually killed (user command)")
            except queue.Empty:
                pass

            time.sleep(0.1)

        if not q.empty():
            status, info = q.get()
            return (status, info)

        return ("err", "Worker exited without reporting status (possible UE crash)")

    finally:
        stop_evt.set()


def main():
    run_idx = XXX  # <<< CHANGED: set here the starting run number (e.g., 193, 220, ...)

    for z in DEPTHS_Z:
        for start_pos in START_POSITIONS:
            for traj in TRAJECTORIES:

                input(
                    f"[{run_idx:04d}] Ready to start the run (z={z}, traj={traj.name})? Press ENTER to continue..."
                )

                attempt = 0
                while True:
                    attempt += 1

                    print(
                        "During the run you can type 'q' + ENTER to interrupt (manual timeout)."
                    )
                    status, info = run_single_with_timeout_or_manual_kill(
                        start_pos, z, traj, run_idx
                    )

                    if status == "ok":
                        print(f"[{run_idx:04d}] OK (attempt {attempt})", flush=True)
                        run_idx += 1
                        time.sleep(1)
                        break

                    print(
                        f"[{run_idx:04d}] FAILED (attempt {attempt}): {status}\n{info}",
                        flush=True
                    )
                    time.sleep(1)

    print("\nDATASET COMPLETE", flush=True)


if __name__ == "__main__":
    main()

