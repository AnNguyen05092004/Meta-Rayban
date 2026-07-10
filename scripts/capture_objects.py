"""
Capture object photos from webcam into data/objects/<label>/.

Usage:
  python -m scripts.capture_objects --name vi_cua_toi --n 10
  python -m scripts.capture_objects --name chia_khoa --out data/objects --n 12

Controls:
  SPACE = save current frame
  ESC   = quit
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    p = argparse.ArgumentParser(description="Capture object images for CPM object personalization.")
    p.add_argument("--name", required=True, help="object label / output folder name")
    p.add_argument("--out", default="data/objects", help="dataset root")
    p.add_argument("--n", type=int, default=10, help="number of images to capture")
    p.add_argument("--cam", type=int, default=0, help="webcam index")
    args = p.parse_args()

    import cv2  # lazy: only needed after argparse succeeds

    outdir = os.path.join(args.out, args.name)
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(
            "Khong mo duoc webcam. Kiem tra quyen camera cua Terminal/VS Code "
            "hoac thu --cam 1 neu may co nhieu camera."
        )

    print(f"Chup object '{args.name}': SPACE=chup, ESC=thoat. Can {args.n} anh.")
    count = 0
    try:
        while count < args.n:
            ok, frame = cap.read()
            if not ok:
                print("Khong doc duoc khung hinh tu webcam.")
                break

            view = frame.copy()
            cv2.putText(
                view,
                f"{args.name}: {count}/{args.n}  SPACE=save  ESC=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("capture object", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                path = os.path.join(outdir, f"{args.name}_{count:02d}.jpg")
                cv2.imwrite(path, frame)
                count += 1
                print(f"  saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done: {count} images in '{outdir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
