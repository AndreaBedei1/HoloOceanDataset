import os
import yaml
import numpy as np
import holoocean
import time

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
OBJECT_CLASS = "seafloor"
MAP_NAME = "DAM"

DEPTHS_M = [60, 55, 50, 45]

FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"
SONAR_KEY = "ImagingSonar"

MAX_FRAMES_PER_RUN = 100

START_X = -300
START_YS = [200, 202, 204] 


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


def run_single(depth_m: float, start_y: float, traj, run_idx: int):

    run_id = f"run_{run_idx:04d}"
    print(f"\nAvvio {run_id} | depth={depth_m} | y={start_y} | motion={traj.name}")

    run_metadata = {
        "run_id": run_id,
        "dataset_version": "v1.0",
        "map": MAP_NAME,
        "primary_object": OBJECT_CLASS,

        "initial_position": {
            "x": START_X,
            "y": start_y,
            "z": -depth_m,
        },
        "initial_depth_m": depth_m,
        "motion_pattern": traj.name,
        "control_mode": "thruster",

        "vertical_motion": {
            "enabled": False,
            "method": "none",
            "vertical_thrust": 0.0,
        },

        "termination": {
            "type": "max_frames",
            "max_frames": MAX_FRAMES_PER_RUN,
            "y_threshold": None,
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

        "notes": "DAM static-depth acquisition",
    }


    run_path = os.path.join(DATASET_ROOT, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    if traj.name == "lateral":
        rot = [0, 0, -360]
    elif traj.name == "lateral_opposite":
        rot = [0, 0, -180]
    else:
        rot = [0, 0, -90]

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

            for key, sensor_name in SENSOR_MAP.items():
                if sensor_name in state:
                    last[key] = state[sensor_name]

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
    print(f"{run_id} completata")


def main():
    run_idx = 0 
    for depth in DEPTHS_M:
        for start_y in START_YS:
            for traj in TRAJECTORIES:
                run_single(depth, start_y, traj, run_idx)
                run_idx += 1
                time.sleep(2)

    print("\n DATASET COMPLETE")


if __name__ == "__main__":
    main()
