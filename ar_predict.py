"""
AR Object Detection with YOLOv5
--------------------------------
This script uses YOLOv5 to perform real-time object detection on a webcam feed
and adds simple AR-style overlays:

- 2D labels on detected objects (class, confidence)
- Basic activity estimation for people (standing / sitting / lying)
- Click-on-object interaction using the mouse
- 3D-style cube drawn on the selected object (simple AR visualization)

This is designed as a teaching/demo script for an Advanced Computer Vision
AR Object Detection project.
"""

import os
import sys
from pathlib import Path

import cv2
import torch

# -------------------- PATH SETUP --------------------

# Current file and project root (assuming this file is in the YOLOv5 folder)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 root if placed there

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# -------------------- YOLOv5 IMPORTS --------------------

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import LoadStreams
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    LOGGER,
    check_imshow,
)


# -------------------- GLOBAL STATE FOR INTERACTION --------------------

# Stores detections for current frame:
# list of dicts: { 'xyxy': [x1,y1,x2,y2], 'cls': int, 'conf': float, 'activity': str }
CURRENT_DETECTIONS = []

# Stores selected detection index (if user clicked an object)
SELECTED_IDX = None


# -------------------- SIMPLE ACTIVITY RECOGNITION --------------------

def estimate_person_activity(xyxy):
    """
    Very simple heuristic-based activity estimation for a person.

    Uses aspect ratio of the bounding box:
    - Tall box  -> standing
    - Medium    -> sitting
    - Wide      -> lying/sleeping

    This is NOT a real activity recognition model,
    but it is enough for a demo for your assignment.
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return "unknown"

    ratio = h / (w + 1e-6)

    if ratio > 1.6:
        return "standing"
    elif ratio > 1.1:
        return "sitting"
    else:
        return "lying/sleeping"


# -------------------- SIMPLE 3D CUBE DRAWING --------------------

def draw_3d_cube(img, xyxy, color=(0, 255, 255), thickness=2):
    """
    Draws a simple 3D-looking cube on top of the bounding box.
    This gives a basic AR-style 3D overlay aligned with the detection.

    xyxy = [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, xyxy)
    h = y2 - y1

    # "Top" rectangle will be slightly above the original bounding box
    offset = int(0.3 * h)  # how tall the cube appears visually

    # Bottom rectangle (original bounding box)
    ptb1 = (x1, y1)
    ptb2 = (x2, y1)
    ptb3 = (x2, y2)
    ptb4 = (x1, y2)

    # Top rectangle (shifted up)
    ptt1 = (x1, y1 - offset)
    ptt2 = (x2, y1 - offset)
    ptt3 = (x2, y2 - offset)
    ptt4 = (x1, y2 - offset)

    # Draw bottom rectangle
    cv2.line(img, ptb1, ptb2, color, thickness)
    cv2.line(img, ptb2, ptb3, color, thickness)
    cv2.line(img, ptb3, ptb4, color, thickness)
    cv2.line(img, ptb4, ptb1, color, thickness)

    # Draw top rectangle
    cv2.line(img, ptt1, ptt2, color, thickness)
    cv2.line(img, ptt2, ptt3, color, thickness)
    cv2.line(img, ptt3, ptt4, color, thickness)
    cv2.line(img, ptt4, ptt1, color, thickness)

    # Connect top and bottom corners
    cv2.line(img, ptb1, ptt1, color, thickness)
    cv2.line(img, ptb2, ptt2, color, thickness)
    cv2.line(img, ptb3, ptt3, color, thickness)
    cv2.line(img, ptb4, ptt4, color, thickness)


# -------------------- MOUSE CALLBACK (INTERACTION) --------------------

def on_mouse(event, x, y, flags, param):
    """
    OpenCV mouse callback. When the user clicks on the image,
    we check if they clicked on any bounding box.

    If yes, we mark that detection as "selected" so we can:
    - Highlight it visually
    - Show 3D cube overlay
    - Show additional info (AR interaction)
    """
    global SELECTED_IDX, CURRENT_DETECTIONS

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check each detection to see if click is inside it
        for idx, det in enumerate(CURRENT_DETECTIONS):
            x1, y1, x2, y2 = det["xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                SELECTED_IDX = idx
                LOGGER.info(
                    f"Selected object: class={det['cls']}, "
                    f"conf={det['conf']:.2f}, activity={det['activity']}"
                )
                return

        # If clicked on background (no box), clear selection
        SELECTED_IDX = None


# -------------------- MAIN AR INFERENCE FUNCTION --------------------

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",    # standard small model
    source="0",                     # webcam by default
    imgsz=(640, 480),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
):

    global CURRENT_DETECTIONS, SELECTED_IDX

    # Select device (CPU/GPU)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, class_names, pt = model.stride, model.names, model.pt

    # Ensure image size is compatible with model stride
    imgsz = check_img_size(imgsz, s=stride)

    # Load webcam stream
    LOGGER.info("Starting webcam stream for AR object detection...")
    view_img = check_imshow()
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # number of webcam sources (usually 1)

    if not view_img:
        LOGGER.warning("cv2.imshow() is not supported on this system.")

    # Set up mouse callback on the main window
    window_name = "AR Object Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess input for the model
        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255.0  # scale 0–255 to 0.0–1.0

        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]  # add batch dimension

        # Inference
        preds = model(im_tensor)

        # NMS (non-max suppression)
        preds = non_max_suppression(
            preds,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            max_det=1000,
        )

        # Handle each stream in the batch (usually just one webcam)
        for i, det in enumerate(preds):
            im0 = im0s[i].copy()

            CURRENT_DETECTIONS = []  # reset detections for this frame

            if len(det):
                # Rescale boxes from model image size to original image size
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                # Loop through detections
                for *xyxy, conf, cls in det[:, :6]:
                    xyxy = [int(v) for v in xyxy]
                    cls = int(cls)
                    conf = float(conf)

                    # Only do activity for 'person' class (COCO class 0)
                    if cls == 0:
                        activity = estimate_person_activity(xyxy)
                    else:
                        activity = "static object"

                    CURRENT_DETECTIONS.append(
                        {"xyxy": xyxy, "cls": cls, "conf": conf, "activity": activity}
                    )

                    # Standard 2D AR overlay (box + label + activity)
                    label = f"{class_names[cls]} {conf:.2f} | {activity}"
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        im0,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # If user selected an object, highlight it and draw 3D cube
            if SELECTED_IDX is not None and 0 <= SELECTED_IDX < len(CURRENT_DETECTIONS):
                sel = CURRENT_DETECTIONS[SELECTED_IDX]
                x1, y1, x2, y2 = sel["xyxy"]

                # Thicker bounding box for selected object
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # 3D cube overlay
                draw_3d_cube(im0, sel["xyxy"], color=(0, 255, 255), thickness=2)

                # Optional: AR info panel at the top-left
                info = f"Selected: {class_names[sel['cls']]} | {sel['activity']} | conf={sel['conf']:.2f}"
                cv2.rectangle(im0, (10, 10), (10 + 430, 45), (0, 0, 0), -1)
                cv2.putText(
                    im0,
                    info,
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Show frame
            if view_img:
                cv2.imshow(window_name, im0)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Quitting AR demo.")
                    return


if __name__ == "__main__":
    # You can adjust weights path if needed, e.g. "runs/train/exp/weights/best.pt"
    run(weights=ROOT / "yolov5s.pt", source="0")
