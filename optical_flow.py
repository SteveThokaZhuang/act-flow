import numpy as np
import os
import cv2
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython

def compute_flow(prev_img, cur_img, raft_model, legacy_mode=False):
    # 统一转为Tensor
    prev_tensor = torch.from_numpy(prev_img).permute(2,0,1).float().cuda() / 255.0
    cur_tensor = torch.from_numpy(cur_img).permute(2,0,1).float().cuda() / 255.0

    with torch.no_grad():
        _, flow_up = raft_model(prev_tensor.unsqueeze(0), cur_tensor.unsqueeze(0))
        flow = flow_up[0].cpu().numpy()

    if legacy_mode:  # 兼容Farneback格式
        flow = flow / 10.0  # 缩放至相似范围
        flow = flow.astype(np.float32)  # 确保精度一致
    
    return flow
def visualize_flow(flow, title):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow(title, bgr)

# visualize_flow(raft_flow, "RAFT")
# visualize_flow(farneback_flow, "Farneback")
# cv2.waitKey(0)
class OpticalFlowAttach():
    def __init__(self):
        self.prev_images = {}
        self.curr_images = {}