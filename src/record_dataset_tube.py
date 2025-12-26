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

DATASET_ROOT = "dataset/runs"
OBJECT_CLASS = "tube"
MAP_NAME = "DAM"
# [60, 55, 50, 45]
DEPTHS_M = [60, 55, 50, 45]

FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"
SONAR_KEY = "ImagingSonar"

MAX_FRAMES_PER_RUN = 100

START_X = 30
START_YS = [42, 44, 46]

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

def run_single(depth_m, start_y, traj, run_idx):

    run_id = f"run_{run_idx:04d}"
    print(f"\n {run_id} | depth={depth_m} | y={start_y} | motion={traj.name}")

    run_metadata = {
        "run_id": run_id,
        "primary_object": OBJECT_CLASS,
        "initial_depth_m": depth_m,
        "initial_position": [START_X, start_y, -depth_m],
        "map": MAP_NAME,
        "motion_pattern": traj.name,
        "notes": "tube, partially buried, always visible in sonar"
    }

    run_path = os.path.join(DATASET_ROOT, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    if traj.name == "lateral":
        rot = [0, 0, -90]
    elif traj.name == "lateral_opposite":
        rot = [0, 0, 90]
    else:
        rot = [0, 0, 180]

    rov = Rover.BlueROV2(
        name="rov0",
        location=[START_X, start_y, -depth_m], 
        rotation=rot,                           
        control_scheme=0,
    )

    scenario = (
        ScenarioConfig("DatasetRun")
        .set_world(World.Dam)
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
        env.water_fog(5.0, 5)

        last = {}
        t = 0

        while writer.frame_id < MAX_FRAMES_PER_RUN:

            state = env.step(traj.command(t))
            t += 1

            for k, s in SENSOR_MAP.items():
                if s in state:
                    last[k] = state[s]

            if SONAR_KEY not in state:
                continue

            telemetry = {
                "pose": parse_pose(last.get("Pose")),
                "velocity": parse_velocity(last.get("Velocity")),
                "altitude": parse_depth(last.get("Depth")),
                "motion": estimate_motion_state(last.get("IMU")),
            }

            writer.write_frame(state, telemetry)

    writer.close()
    print(f" {run_id} compelte")

def main():
    run_idx = 48
    for depth in DEPTHS_M:
        for start_y in START_YS:               
            for traj in TRAJECTORIES:
                run_single(depth, start_y, traj, run_idx) 
                run_idx += 1

    print("\n DATASET COMPLETE ")


if __name__ == "__main__":
    main()
