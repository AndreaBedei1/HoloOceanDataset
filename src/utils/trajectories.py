import numpy as np


# ============================
# TRAJECTORY CONTROLLERS
# ============================
class ForwardTrajectory:
    name = "forward"

    def __init__(self, speed=25.0):
        self.speed = float(speed)

    def command(self, t):
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4:8] = self.speed
        return cmd


class LateralTrajectory:
    name = "lateral"

    def __init__(self, speed=25.0):
        self.speed = float(speed)

    def command(self, t):
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4] += self.speed
        cmd[5] -= self.speed
        cmd[6] += self.speed
        cmd[7] -= self.speed

        return cmd


class ZigZagTrajectory:
    """
    Moves forward with lateral zig-zag (alternating strafe)
    """
    name = "zigzag"

    def __init__(self, speed=20.0, sway=15.0, period=50):
        self.speed = float(speed)
        self.sway = float(sway)
        self.period = int(period)

    def command(self, t):
        cmd = np.zeros(8, dtype=np.float32)
        cmd[4:8] += self.speed

        phase = (t // self.period) % 2

        if phase == 0:
            cmd[4] += self.sway
            cmd[5] -= self.sway
            cmd[6] += self.sway
            cmd[7] -= self.sway
        else:
            cmd[4] -= self.sway
            cmd[5] += self.sway
            cmd[6] -= self.sway
            cmd[7] += self.sway

        return cmd