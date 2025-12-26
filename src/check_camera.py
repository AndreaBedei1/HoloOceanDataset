import glob
import cv2

files = sorted(glob.glob(
    "dataset/runs/run_0003/front_rgb/*.jpg"
))

cv2.namedWindow("RGB Viewer", cv2.WINDOW_NORMAL)

for f in files:
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Errore nel caricare {f}")
        continue

    cv2.imshow("RGB Viewer", img)

    key = cv2.waitKey(200)
    if key == 27: 
        break

cv2.destroyAllWindows()
