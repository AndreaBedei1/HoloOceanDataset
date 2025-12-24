import os
import yaml
import numpy as np
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
    ZigZagTrajectory,
)



# ============================
# CONFIG DATASET
# ============================

DATASET_ROOT = "dataset/runs"
OBJECT_CLASS = "seafloor"
MAP_NAME = "DAM"

DEPTHS_M = [60, 55, 50, 45]

FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"
SONAR_KEY = "ImagingSonar"

MAX_FRAMES_PER_RUN = 100




TRAJECTORIES = [                              
    ForwardTrajectory(),
    LateralTrajectory(),
    ZigZagTrajectory(),
]


# ============================
# SENSOR CACHE MAP
# ============================

SENSOR_MAP = {
    "Pose": "PoseSensor",
    "Velocity": "VelocitySensor",
    "IMU": "IMUSensor",
    "Depth": "DepthSensor",
}


# ============================
# SINGLE RUN
# ============================

def run_single(depth_m: float, traj, run_idx: int): 

    run_id = f"run_{run_idx:04d}"
    print(f"\nAvvio {run_id} | depth={depth_m} | motion={traj.name}")  

    run_metadata = {
        "run_id": run_id,
        "primary_object": OBJECT_CLASS,
        "initial_depth_m": depth_m,
        "map": MAP_NAME,
        "motion": traj.name,                           
        "notes": "Seafloor-only acquisition"
    }

    run_path = os.path.join(DATASET_ROOT, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    # POSIZIONE DI PARTENZA: INVARIATA
    rov = Rover.BlueROV2(
        name="rov0",
        location=[-300, 200, -depth_m],      
        rotation=[0, 0, -90] if traj.name == "lateral" else [0, 0, -90],  
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


# ============================
# MAIN
# ============================

def main():
    run_idx = 0
    for depth in DEPTHS_M:                    
        for traj in TRAJECTORIES:             
            run_single(depth, traj, run_idx)
            run_idx += 1

    print("\nDataset complete")


if __name__ == "__main__":
    main()
