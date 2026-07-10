"""
Chụp nhanh ảnh dataset từ webcam vào data/<out>/<name>/.

Dùng để tạo bộ ảnh test thật (khuôn mặt hoặc đồ vật).
    python -m scripts.capture_faces --name Lan --n 10
    python -m scripts.capture_faces --name vi_cua_toi --out data/objects --n 10

Điều khiển: PHÍM CÁCH = chụp 1 ảnh, ESC = thoát.
Lưu ý macOS: lần đầu cần cấp quyền camera cho Terminal/VS Code
(System Settings → Privacy & Security → Camera).
"""

from __future__ import annotations

import argparse
import os


def main():
    import cv2  # lazy: chỉ cần khi thực sự chạy

    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="tên nhãn (tên thư mục con)")
    p.add_argument("--out", default="data/faces", help="thư mục gốc dataset")
    p.add_argument("--n", type=int, default=10, help="số ảnh cần chụp")
    p.add_argument("--cam", type=int, default=0, help="chỉ số webcam")
    args = p.parse_args()

    outdir = os.path.join(args.out, args.name)
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Không mở được webcam. Kiểm tra quyền camera của Terminal/VS Code trên macOS.")

    print(f"Chụp cho '{args.name}': [PHÍM CÁCH]=chụp, [ESC]=thoát. Cần {args.n} ảnh.")
    count = 0
    while count < args.n:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được khung hình.")
            break
        view = frame.copy()
        cv2.putText(
            view,
            f"{args.name}: {count}/{args.n}  [SPACE]=chup  [ESC]=thoat",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("capture (nhan SPACE de chup)", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == 32:  # SPACE
            path = os.path.join(outdir, f"{args.name}_{count:02d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  đã lưu {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Xong: {count} ảnh trong '{outdir}'")


if __name__ == "__main__":
    main()
