import os
import yaml
import holoocean

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

DEPTHS_Z = [-285.0, -280.0, -275.0, -270.0]

STOP_Y = -52.0
DESCENT_PER_FRAME = -0.02  # meters per frame (gentle descent)

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
    "Depth": "DepthSensor",
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
        "primary_object": OBJECT_CLASS,
        "initial_depth_m": start_z,
        "initial_position": [x0, y0, start_z],
        "map": MAP_NAME,
        "motion_pattern": traj.name,
        "descent_per_frame": DESCENT_PER_FRAME,
        "stop_condition": f"y < {STOP_Y}",
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
        show_viewport=True,
        ticks_per_sec=30,
        frames_per_sec=True
    ) as env:

        env.tick(2)

        last = {}
        t = 0

        while True:

            cmd = traj.command(t)
            cmd[2] += DESCENT_PER_FRAME  # apply gentle descent
            t += 1

            state = env.step(cmd)

            for k, s in SENSOR_MAP.items():
                if s in state:
                    last[k] = state[s]

            pose = parse_pose(last.get("Pose"))
            if pose is None:
                continue

            _, y, _ = pose["position"]

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


def main():
    run_idx = 49
    for start_pos in START_POSITIONS:
        for z in DEPTHS_Z:
            for traj in TRAJECTORIES:
                run_single(start_pos, z, traj, run_idx)
                run_idx += 1

    print("\n DATASET COMPLETE")


if __name__ == "__main__":
    main()
