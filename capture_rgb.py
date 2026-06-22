import argparse
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview and capture a USB RGB camera frame.")
    parser.add_argument("--index", type=int, default=1, help="OpenCV camera index.")
    parser.add_argument("--out", type=Path, default=Path("captures/rgb.jpg"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action="store_true", help="Show live preview until q/ESC/s.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Capture after N seconds without preview.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame = None
    try:
        if args.preview:
            print("Preview controls: s = save, q/ESC = quit")
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera opened, but no frame was received.")
                cv2.imshow(f"Camera index {args.index}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("s"):
                    break
        else:
            if args.seconds > 0:
                time.sleep(args.seconds)
            for _ in range(10):
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera opened, but no frame was received.")

        if frame is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out), frame)
            print(f"saved {args.out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
