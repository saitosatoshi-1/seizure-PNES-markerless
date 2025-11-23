import cv2, os, argparse
from ultralytics import YOLO

def convert_to_qt_compatible(in_path, out_path):
    os.system(
        f"ffmpeg -y -i {in_path} -vcodec libx264 -pix_fmt yuv420p "
        f"-profile:v baseline -level 3.0 -movflags +faststart {out_path}"
    )

def process_segmentation(input_video, output_video):
    seg_model = YOLO('yolo11s-seg.pt')

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCaptureが開けません: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = seg_model(frame)
        vis = results[0].plot()
        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--output", default="output.mp4", help="Output video")
    args = parser.parse_args()

    tmp = "tmp_output.mp4"
    process_segmentation(args.input, tmp)

    convert_to_qt_compatible(tmp, args.output)
    print(f"Done. Output saved at: {args.output}")

if __name__ == "__main__":
    main()
