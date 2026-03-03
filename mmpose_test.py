import cv2
import time
import torch
import numpy as np
import os
import mmdet
import mmpose
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import register_all_modules, adapt_mmdet_pipeline
from mmpose.visualization import PoseLocalVisualizer

register_all_modules()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

mmdet_root = os.path.dirname(mmdet.__file__)
mmpose_root = os.path.dirname(mmpose.__file__)

DET_CFG = os.path.join(mmdet_root, '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py')
DET_CKPT = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

POSE_CFG = os.path.join(mmpose_root, '.mim/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py')
POSE_CKPT = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth'

def run():
    print("Loading models...")
    det_model = init_detector(DET_CFG, DET_CKPT, device=DEVICE)
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
    pose_model = init_model(POSE_CFG, POSE_CKPT, device=DEVICE)

    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(pose_model.dataset_meta)

    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("error opening webcam")
        return

    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        det_res = inference_detector(det_model, frame)
        pred_instances = det_res.pred_instances.cpu().numpy()
        bboxes = pred_instances.bboxes[pred_instances.labels == 0]
        scores = pred_instances.scores[pred_instances.labels == 0]

        if len(bboxes) > 0:
            best_idx = np.argmax(scores)
            bbox = bboxes[best_idx:best_idx+1]
            pose_results = inference_topdown(pose_model, frame, bbox)
            visualizer.add_datasample(
                'result',
                frame,
                data_sample=pose_results[0],
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                kpt_thr=0.3
            )
            frame = visualizer.get_image()
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MMPose', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
