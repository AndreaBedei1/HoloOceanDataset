import os
import yaml
import holoocean
import time
import multiprocessing as mp
import traceback

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
OBJECT_CLASS = "submarine"

MAP_NAME = "World.OpenWater"

START_POSITIONS = [
    (15.0, -15.0, -285.0),
    (13.8, -15.0, -285.0),
    (12.5, -15.0, -285.0),
]
# [-285.0, -280.0, -275.0, -270.0]
DEPTHS_Z = [-285.0, -280.0, -275.0, -270.0]

STOP_Y = -53.0
DESCENT_PER_FRAME = 4  # meters per frame (gentle descent)

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
        return [0, 0, -358]
    elif traj.name == "lateral_opposite":
        return [0, 0, -178]
    else:
        return [0, 0, -88]

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
            "enabled": True,
            "method": "vertical_thrusters",
            "vertical_thrust": DESCENT_PER_FRAME,
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

        "notes": "OpenWater descending run",
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
        .set_world(World.OpenWater)
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
        frames_per_sec=True
    ) as env:

        env.tick(2)
        env.water_fog(5.0, 5)

        last = {}
        t = 0

        while True:

            cmd = traj.command(t)
            cmd[0:4] = -DESCENT_PER_FRAME  # apply gentle descent
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



RUN_TIMEOUT_SEC = 90  # ad es. 1.3 minuti per run (scegli tu)
def _worker_run_single(args, q):
    start_pos, start_z, traj, run_idx = args
    try:
        run_single(start_pos, start_z, traj, run_idx)
        q.put(("ok", None))
    except Exception:
        q.put(("err", traceback.format_exc()))

def run_single_with_timeout(start_pos, start_z, traj, run_idx, timeout_sec=RUN_TIMEOUT_SEC):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_run_single, args=((start_pos, start_z, traj, run_idx), q))
    p.start()

    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(10)
        return ("timeout", f"Run {run_idx} killed after {timeout_sec}s")

    if not q.empty():
        status, info = q.get()
        return (status, info)

    return ("err", "Worker exited without reporting status (possible UE crash)")

def main():
    # 97
    run_idx = 97
    for z in DEPTHS_Z:
        for start_pos in START_POSITIONS:
            for traj in TRAJECTORIES:

                attempt = 0
                while True:
                    attempt += 1

                    status, info = run_single_with_timeout(start_pos, z, traj, run_idx)

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

