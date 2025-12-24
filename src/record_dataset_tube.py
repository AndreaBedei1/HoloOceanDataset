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


# ============================
# CONFIG DATASET
# ============================

DATASET_ROOT = "dataset/runs"
OBJECT_CLASS = "tube"
MAP_NAME = "DAM"

DEPTHS_M = [60, 55, 50, 45]

FRONT_CAM = "FrontCamera"
BOTTOM_CAM = "SonarCamera"
SONAR_KEY = "ImagingSonar"

MAX_FRAMES_PER_RUN = 100

# ============================
# TRAJECTORY CONTROLLERS
# ============================

class ForwardTrajectory:
    name = "forward"
    yaw_deg = 180

    def __init__(self, speed=25.0):
        self.speed = float(speed)

    def command(self, t: int) -> np.ndarray:
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4:] = self.speed  # surge
        return cmd

class LateralTrajectory:
    name = "lateral"

    def __init__(self, speed=25.0, yaw_rate=20.0, turn_frames=30):
        self.speed = speed
        self.yaw_rate = yaw_rate
        self.turn_frames = turn_frames

    def command(self, t):
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4] -= self.speed
        cmd[5] += self.speed
        cmd[6] -= self.speed
        cmd[7] += self.speed
        return cmd



class ZigZagTrajectory:
    """
    Avanza in forward con zig-zag laterale (strafe alternato)
    """
    name = "zigzag"

    def __init__(self, speed=20.0, sway=15.0, period=50):
        self.speed = speed
        self.sway = sway
        self.period = period

    def command(self, t):
        cmd = np.zeros(8, dtype=np.float32)

        # --- forward puro (W) ---
        cmd[4:8] += self.speed

        # --- zigzag laterale (A / D) ---
        phase = (t // self.period) % 2

        if phase == 0:
            # strafe left (A)
            cmd[4] += self.sway
            cmd[5] -= self.sway
            cmd[6] += self.sway
            cmd[7] -= self.sway
        else:
            # strafe right (D)
            cmd[4] -= self.sway
            cmd[5] += self.sway
            cmd[6] -= self.sway
            cmd[7] += self.sway

        return cmd


TRAJECTORIES = [
    ForwardTrajectory(),
    LateralTrajectory(),
    ZigZagTrajectory(),
]

# ============================
# SENSOR MAP
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

def run_single(depth_m, traj, run_idx):

    run_id = f"run_{run_idx:04d}"
    print(f"\n {run_id} | depth={depth_m} | motion={traj.name}")

    # ---------- METADATA ----------
    run_metadata = {
        "run_id": run_id,
        "primary_object": OBJECT_CLASS,
        "initial_depth_m": depth_m,
        "map": MAP_NAME,
        "motion_pattern": traj.name,
        "notes": "tube, partially buried, always visible in sonar"
    }

    run_path = os.path.join(DATASET_ROOT, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.safe_dump(run_metadata, f)

    # ---------- ROVER ----------
    rov = Rover.BlueROV2(
        name="rov0",
        location=[30, 45, -depth_m],
        rotation = [0, 0, 180] if traj.name != "lateral" else [0, 0, -90],
        control_scheme=0,
    )

    scenario = (
        ScenarioConfig("DatasetRun")
        .set_world(World.Dam)
        .add_agent(rov)
    )

    # ---------- WRITER ----------
    writer = DatasetWriter(
        root=DATASET_ROOT,
        run_id=run_id,
        front_cam_key=FRONT_CAM,
        bottom_cam_key=BOTTOM_CAM,
        sonar_key=SONAR_KEY,
        pose_to_csv_fields=pose_to_csv_fields,
        velocity_to_csv_fields=velocity_to_csv_fields,
    )

    # ---------- SIM ----------
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
    print(f" {run_id} completata")


# ============================
# MAIN
# ============================

def main():
    run_idx = 12
    # for depth in DEPTHS_M:
    #     for traj in TRAJECTORIES:
    #         run_single(depth, traj, run_idx)
    #         run_idx += 1
    run_single(DEPTHS_M[0], ZigZagTrajectory(), run_idx)

    print("\n DATASET COMPLETO")


if __name__ == "__main__":
    main()
