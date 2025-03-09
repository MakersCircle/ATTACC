

def get_frame_obj_depth(frame_depth, frame_obj_detections):
    obj_depths = []
    for x1, y1, x2, y2, _, _ in frame_obj_detections:
        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
        obj_depths.append(frame_depth[cy][cx].item())
    return obj_depths