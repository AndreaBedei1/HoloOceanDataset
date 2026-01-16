import glob
import time
import numpy as np
import os
from utils.sonar_viz import PolarSonarVisualizerAsync
#193
files = sorted(glob.glob(
    "dataset/runs/run_0288/sonar_raw/*.npz"
))

viz = PolarSonarVisualizerAsync(
    azimuth_deg=90,
    range_min=1.0,
    range_max=30.0,
    plot_hz=5,
    use_cuda=True
)

text_handle = None  # handle del testo overlay

try:
    for f in files:
        sonar = np.load(f)["intensity"]

        viz.submit(sonar)
        viz.update_plot()
        image_id = os.path.basename(f)

        if text_handle is None:
            text_handle = viz.ax.text(
                0.02, 0.95,
                image_id,
                transform=viz.ax.transAxes,
                color="white",
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=3)
            )
        else:
            text_handle.set_text(image_id)

        time.sleep(1)

finally:
    viz.close()
