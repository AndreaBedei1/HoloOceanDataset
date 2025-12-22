import os
import csv
import time
import cv2
import numpy as np


class DatasetWriter:
    """
    Handles saving of:
    - RGB images (front + bottom)
    - raw sonar (.npz)
    - telemetry CSV (pose, velocity, altitude)
    """

    def __init__(
        self,
        root: str,
        run_id: str,
        front_cam_key: str,
        bottom_cam_key: str,
        sonar_key: str,
        pose_to_csv_fields,
        velocity_to_csv_fields,
    ):
        """
        Parameters
        ----------
        root : str
            Dataset root directory (e.g. dataset/runs)
        run_id : str
            Run identifier (e.g. run_0001)
        front_cam_key : str
            Key used in HoloOcean state for front camera
        bottom_cam_key : str
            Key used in HoloOcean state for bottom camera
        sonar_key : str
            Key used in HoloOcean state for sonar
        pose_to_csv_fields : callable
            Function converting pose dict -> [x,y,z,qx,qy,qz,qw]
        velocity_to_csv_fields : callable
            Function converting velocity dict -> [vx,vy,vz]
        """

        self.front_cam_key = front_cam_key
        self.bottom_cam_key = bottom_cam_key
        self.sonar_key = sonar_key
        self.pose_to_csv_fields = pose_to_csv_fields
        self.velocity_to_csv_fields = velocity_to_csv_fields

        self.run_path = os.path.join(root, run_id)
        self.frame_id = 0

        # directories
        self.front_dir = os.path.join(self.run_path, "images", "front")
        self.bottom_dir = os.path.join(self.run_path, "images", "bottom")
        self.sonar_dir = os.path.join(self.run_path, "sonar_raw")

        os.makedirs(self.front_dir, exist_ok=True)
        os.makedirs(self.bottom_dir, exist_ok=True)
        os.makedirs(self.sonar_dir, exist_ok=True)

        # CSV
        self.csv_file = open(
            os.path.join(self.run_path, "telemetry.csv"),
            "w",
            newline=""
        )
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ns",
            "frame_id",
            "pose_x", "pose_y", "pose_z",
            "qx", "qy", "qz", "qw",
            "vel_x", "vel_y", "vel_z",
            "altitude",
            "front_img",
            "bottom_img",
            "sonar_raw",
        ])

    # -------------------------------------------------

    def write_frame(self, state: dict, telemetry: dict):
        """
        Writes one dataset frame.
        A frame is valid ONLY if pose and velocity are available.
        """

        fid = f"{self.frame_id:06d}"
        ts = time.time_ns()

        # ============================
        # IMAGES
        # ============================

        front_path = ""
        if self.front_cam_key in state:
            front_path = f"images/front/{fid}.png"
            cv2.imwrite(
                os.path.join(self.run_path, front_path),
                state[self.front_cam_key][:, :, :3]
            )

        bottom_path = ""
        if self.bottom_cam_key in state:
            bottom_path = f"images/bottom/{fid}.png"
            cv2.imwrite(
                os.path.join(self.run_path, bottom_path),
                state[self.bottom_cam_key][:, :, :3]
            )

        # ============================
        # SONAR RAW
        # ============================

        sonar_path = ""
        if self.sonar_key in state:
            sonar = state[self.sonar_key]
            sonar_path = f"sonar_raw/{fid}.npz"

            np.savez(
                os.path.join(self.run_path, sonar_path),
                intensity=sonar,
                timestamp_ns=ts,
                frame_id=fid,
            )

        # ============================
        # TELEMETRY
        # ============================

        pose_fields = self.pose_to_csv_fields(telemetry.get("pose"))
        vel_fields = self.velocity_to_csv_fields(telemetry.get("velocity"))
        altitude = telemetry.get("altitude")

        if pose_fields is None or vel_fields is None:
            return  # skip invalid frame

        self.csv_writer.writerow([
            ts,
            fid,
            *pose_fields,
            *vel_fields,
            altitude,
            front_path,
            bottom_path,
            sonar_path,
        ])

        self.frame_id += 1

    # -------------------------------------------------

    def close(self):
        self.csv_file.close()
