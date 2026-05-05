import cv2
import numpy as np
from vlm.coco_classes import COCO_CLASSES
from vlm.detector.yolov7 import YOLOv7Client
from vlm.segmentor.sam import MobileSAMClient
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.utils.get_itm_message import get_itm_message

yolov7_detector = YOLOv7Client()
blip2_itm = BLIP2ITMClient()
sam_segmentor = MobileSAMClient()
dino_detector = GroundingDINOClient()


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            color,
            2,
        )
        cv2.putText(
            segmented_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    return segmented_img, object_mask

def get_object(right_label, img, cfg, similar_answer, return_metadata=False):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []
    landmark_detections = []
    coco_label = []
    dino_label = []
    right_label_list = list(map(str.strip, right_label.split('|')))
    # print(f"right_label_list: {right_label_list}")
    all_answer = right_label_list + similar_answer
    landmark_labels = _configured_landmark_labels(cfg)
    publish_landmarks_to_map = bool(_cfg_get(cfg, "publish_landmarks_to_map", False))
    max_landmarks = int(_cfg_get(cfg, "max_landmark_detections", 12) or 12)
    for label in all_answer:
        if label in COCO_CLASSES:
            coco_label.append(label)
        else:
            dino_label.append(label)
    yolo_landmark_labels = [
        label for label in landmark_labels if label in COCO_CLASSES and label not in all_answer
    ]

    if any(item in dino_label for item in right_label_list):
        dino_label = all_answer
        coco_label = []
        for label in right_label_list:
            if label in COCO_CLASSES:
                coco_label.append(label)

    if coco_label or yolo_landmark_labels:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms, 
                                            conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)
            elif detections.phrases[idx] in coco_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(list(all_answer).index(label_detected) - len(right_label_list)+1)
            elif label_detected in yolo_landmark_labels:
                landmark_detections.append(
                    _detection_metadata(
                        idx=idx,
                        detections=detections,
                        img=img,
                        label=label_detected,
                        score=score,
                        source="yolov7_landmark",
                    )
                )
                if publish_landmarks_to_map:
                    segmented_img, object_mask = get_segmentation(
                        segmented_img, idx, detections, img, label_detected, score, color=(0, 180, 255)
                    )
                    score_list.append(score)
                    object_masks_list.append(object_mask)
                    label_list.append(1000 + yolo_landmark_labels.index(label_detected))
                if len(landmark_detections) >= max_landmarks:
                    break

    if dino_label:
        caption = ' '.join(f'{item}.  ' for item in dino_label)
        detections = dino_detector.predict(img, caption=caption, 
                                        box_threshold=cfg.groundingDINO.confidence_threshold_dino, text_threshold=cfg.groundingDINO.text_threshold)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if label_detected in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)

            elif label_detected in dino_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(0, 255, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(list(all_answer).index(label_detected) - len(right_label_list)+1)

    landmark_detections = sorted(
        landmark_detections, key=lambda item: item.get("confidence") or 0.0, reverse=True
    )[:max_landmarks]
    if return_metadata:
        return segmented_img, score_list, object_masks_list, label_list, landmark_detections
    return segmented_img, score_list, object_masks_list, label_list


def _configured_landmark_labels(cfg):
    if not bool(_cfg_get(cfg, "enable_landmark_detection", True)):
        return []
    labels = _cfg_get(cfg, "landmark_labels", [])
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [item.strip() for item in labels.split(",")]
    return [str(item).strip() for item in labels if str(item).strip()]


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _detection_metadata(idx, detections, img, label, score, source):
    box = detections.boxes[idx]
    if hasattr(box, "detach"):
        box = box.detach().cpu().numpy()
    box = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = box.tolist()
    width = float(img.shape[1])
    height = float(img.shape[0])
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return {
        "id": int(idx),
        "label": str(label),
        "confidence": float(score),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "center": [float(center_x), float(center_y)],
        "direction": _horizontal_direction(center_x),
        "area_ratio": float(max(0.0, (x2 - x1) * (y2 - y1))),
        "source": source,
        "is_landmark": True,
        "is_target_candidate": False,
        "grounded_in_current_observation": True,
        "image_size": [int(width), int(height)],
    }


def _horizontal_direction(center_x):
    if center_x < 0.33:
        return "left"
    if center_x > 0.67:
        return "right"
    return "center"

def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()
    if label in COCO_CLASSES:
        detections = yolov7_detector.predict(img, agnostic_nms=cfg.yolo.agnostic_nms,
                                             conf_thres=cfg.yolo.confidence_threshold_yolo, iou_thres=cfg.yolo.iou_threshold_yolo)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] == label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine:.3f}, itm_score: {itm_score:.3f}")
                score_list.append(score)
                object_masks_list.append(object_mask)
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    else:
        detections = dino_detector.predict(img, caption=label, 
                                           box_threshold=cfg.groundingDINO.confidence_threshold_dino, text_threshold=cfg.groundingDINO.text_threshold)
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if score > cfg.groundingDINO.confidence_threshold_dino:
                segmented_img, object_mask = get_segmentation(
                    segmented_img, idx, detections, img, label_detected, score, color=(255, 0, 0)
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine}, itm_score: {itm_score}")
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)
    
    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # Get bounding box coordinates in [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # Expand the box outward; clamp to image boundaries
    x_min = max(int(x_min*(1-expand_pixels)), 0)
    y_min = max(int(y_min*(1-expand_pixels)), 0)
    x_max = min(int(x_max*(1+expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max*(1+expand_pixels)), img.shape[0] - 1)

    # Crop the image to keep only the box region
    img_detected = img[y_min:y_max+1, x_min:x_max+1]

    return img_detected
