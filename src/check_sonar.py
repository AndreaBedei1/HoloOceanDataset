import glob
import time
import numpy as np
from utils.sonar_viz import PolarSonarVisualizerAsync

files = sorted(glob.glob(
    "dataset/runs/run_0001/sonar_raw/*.npz"
))

viz = PolarSonarVisualizerAsync(
    azimuth_deg=90,
    range_min=1.0,
    range_max=30.0,
    plot_hz=5,
    use_cuda=True
)

try:
    for f in files:
        sonar = np.load(f)["intensity"]
        viz.submit(sonar)
        viz.update_plot()
        time.sleep(0.2)
finally:
    viz.close()
