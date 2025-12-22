import os
import yaml
import numpy as np
import holoocean

from lib.scenario_builder import ScenarioConfig
from lib.worlds import World
from lib.rover import Rover

from telemetry.parsing import parse_pose
from telemetry.estimation import (
    estimate_velocity,
    parse_depth,
    estimate_motion_state,
    estimate_front_obstacle,
)

from utils.convert import pose_to_csv_fields
from utils.convert import velocity_to_csv_fields

from utils.writer import DatasetWriter


# ============================
# CONFIG
# ============================

DATASET_ROOT = "dataset/runs"
RUN_ID = "run_0001"
OBJECT_CLASS = "seafloor"  
ALTITUDE_M = 2

SONAR_KEY = "ImagingSonar"
FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"

# ============================
# TRAJECTORY CONTROLLER
# ============================

class ForwardTrajectory:
    def __init__(self, speed=25.0):
        self.speed = speed

    def command(self):
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4:] = self.speed
        return cmd


SENSOR_MAP = {
    "Pose": "PoseSensor",
    "Velocity": "VelocitySensor",
    "IMU": "IMUSensor",
    "DVL": "DVLSensor",
    "RangeFinder": "RangeFinderSensor",
    "Collision": "CollisionSensor",
}


def main():

    run_metadata = {
        "run_id": RUN_ID,
        "primary_object": OBJECT_CLASS,
        "altitude_m": ALTITUDE_M,
        "motion": "forward",
        "notes": "Seafloor-only acquisition"
    }

    run_path = os.path.join(DATASET_ROOT, RUN_ID)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    rov = Rover.BlueROV2(
        name="rov0",
        location=[100, -100, -ALTITUDE_M],
        rotation=[0, 0, 0],
        control_scheme=0,
    )

    scenario = (
        ScenarioConfig("DatasetRun")
        .set_world(World.Dam)
        .add_agent(rov)
    )

    writer = DatasetWriter(
        root=DATASET_ROOT,
        run_id=RUN_ID,
        front_cam_key=FRONT_CAM,
        bottom_cam_key=BOTTOM_CAM,
        sonar_key=SONAR_KEY,
        pose_to_csv_fields=pose_to_csv_fields,
        velocity_to_csv_fields=velocity_to_csv_fields,
    )

    traj = ForwardTrajectory()

    print("Avvio acquisizione dataset...")

    with holoocean.make(
        scenario_cfg=scenario.to_dict(),
        show_viewport=True,
        ticks_per_sec=30,
        frames_per_sec=True
    ) as env:

        last = {}

        while True:
            state = env.step(traj.command())

            for k, v in SENSOR_MAP.items():
                if v in state:
                    last[k] = state[v]

            if SONAR_KEY not in state:
                continue 

            telemetry = {
                "pose": parse_pose(last.get("Pose")),
                "velocity": estimate_velocity(last.get("Velocity")),
                "altitude": parse_depth(last.get("Depth")),
                "motion": estimate_motion_state(last.get("IMU")),
                "front_range": estimate_front_obstacle(last.get("RangeFinder")),
            }

            writer.write_frame(state, telemetry)

            if writer.frame_id >= 300:
                break

    writer.close()
    print("Run completata")


if __name__ == "__main__":
    main()
