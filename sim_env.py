# import numpy as np
# import os
# import cv2
# import collections
# import matplotlib.pyplot as plt
# import torch
# import sys
# from RAFT.core.raft import RAFT
# from dm_control import mujoco
# from dm_control.rl import control
# from dm_control.suite import base

# from constants import DT, XML_DIR, START_ARM_POSE
# from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
# from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
# from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
# from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
# # from optical_flow import OpticalFlowAttach
# sys.path.append('RAFT/core') # raft
# import IPython
# e = IPython.embed
# BOX_POSE = [None] # to be changed from outside

# def make_sim_env(task_name):
#     """
#     Environment for simulated robot bi-manual manipulation, with joint position control
#     Action space:      [left_arm_qpos (6),             # absolute joint position
#                         left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
#                         right_arm_qpos (6),            # absolute joint position
#                         right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

#     Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
#                                         left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
#                                         right_arm_qpos (6),         # absolute joint position
#                                         right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
#                         "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
#                                         left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
#                                         right_arm_qvel (6),         # absolute joint velocity (rad)
#                                         right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
#                         "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
#     """
#     if 'sim_transfer_cube' in task_name:
#         xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
#         physics = mujoco.Physics.from_xml_path(xml_path)
#         task = TransferCubeTask(random=False)
#         env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
#                                   n_sub_steps=None, flat_observation=False)
#     elif 'sim_insertion' in task_name:
#         xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
#         physics = mujoco.Physics.from_xml_path(xml_path)
#         task = InsertionTask(random=False)
#         env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
#                                   n_sub_steps=None, flat_observation=False)
#     else:
#         raise NotImplementedError
#     return env

# class BimanualViperXTask(base.Task):
#     def __init__(self, random=None):
#         super().__init__(random=random)
#         # self.flow = OpticalFlowAttach()
#         self.prev_image = {}
#         self.raft_model = torch.jit.load('raft_weights.pth').to('cuda').eval()  # 预训练权重路径

#     def before_step(self, action, physics):
#         left_arm_action = action[:6]
#         right_arm_action = action[7:7+6]
#         normalized_left_gripper_action = action[6]
#         normalized_right_gripper_action = action[7+6]

#         left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
#         right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

#         full_left_gripper_action = [left_gripper_action, -left_gripper_action]
#         full_right_gripper_action = [right_gripper_action, -right_gripper_action]

#         env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
#         super().before_step(env_action, physics)
#         return

#     def initialize_episode(self, physics):
#         """Sets the state of the environment at the start of each episode."""
#         super().initialize_episode(physics)
#         # TODO: manipulate the optical flow of the camera
#         for cam in ['top', 'left_wrist', 'right_wrist']:
#             img = physics.render(height=480, width=640, camera_id=cam)
#             self.prev_image[cam] = img.copy()
#             # self.flow.prev_images[cam] = img.copy()
#             # self.flow.curr_images[cam] = img.copy()

#     @staticmethod
#     def get_qpos(physics):
#         qpos_raw = physics.data.qpos.copy()
#         left_qpos_raw = qpos_raw[:8]
#         right_qpos_raw = qpos_raw[8:16]
#         left_arm_qpos = left_qpos_raw[:6]
#         right_arm_qpos = right_qpos_raw[:6]
#         left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
#         right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
#         return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

#     @staticmethod
#     def get_qvel(physics):
#         qvel_raw = physics.data.qvel.copy()
#         left_qvel_raw = qvel_raw[:8]
#         right_qvel_raw = qvel_raw[8:16]
#         left_arm_qvel = left_qvel_raw[:6]
#         right_arm_qvel = right_qvel_raw[:6]
#         left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
#         right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
#         return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

#     @staticmethod
#     def get_env_state(physics):
#         raise NotImplementedError

#     def get_observation(self, physics):
#         obs = collections.OrderedDict()
#         obs['qpos'] = self.get_qpos(physics)
#         obs['qvel'] = self.get_qvel(physics)
#         obs['env_state'] = self.get_env_state(physics)
#         obs['images'] = dict()
#         # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
#         # obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
#         # obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
#         # # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
#         # # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
#         for cam in ['top', 'left_wrist', 'right_wrist']:
#             obs['images'][cam] = physics.render(height=480, width=640, camera_id=cam)

#         # ## Optical Flow
#         # obs['flows'] = {}
#         # for cam, cur_img in obs['images'].items():
#         #     prev_img = self.prev_images[cam]
#         #     prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
#         #     cur_gray  = cv2.cvtColor(cur_img,  cv2.COLOR_BGR2GRAY)
#         #     # Farneback
#         #     flow = cv2.calcOpticalFlowFarneback(
#         #         prev_gray, cur_gray,
#         #         None,
#         #         pyr_scale=0.5, levels=3,
#         #         winsize=15, iterations=3,
#         #         poly_n=5, poly_sigma=1.2, flags=0
#         #     )  # flow: (h, w, 2)
#         #     obs['flows'][cam] = flow
#         #     # Update previous image
#         #     self.prev_images[cam] = cur_img.copy()
#         # 使用RAFT计算光流
#         obs['flows'] = {}
#         for cam, cur_img in obs['images'].items():
#             prev_img = self.prev_images.get(cam, cur_img.copy())
            
#             # 将图像转为PyTorch Tensor并归一化
#             prev_tensor = torch.from_numpy(prev_img).permute(2,0,1).float().to('cuda') / 255.0
#             cur_tensor = torch.from_numpy(cur_img).permute(2,0,1).float().to('cuda') / 255.0
            
#             # RAFT计算光流
#             with torch.no_grad():
#                 flow = self.raft_model(prev_tensor.unsqueeze(0), 
#                                     cur_tensor.unsqueeze(0))[0].cpu().numpy()  # [H,W,2]
            
#             obs['flows'][cam] = flow
#             self.prev_images[cam] = cur_img.copy()

#             return obs

#     def get_reward(self, physics):
#         # return whether left gripper is holding the box
#         raise NotImplementedError


# class TransferCubeTask(BimanualViperXTask):
#     def __init__(self, random=None):
#         super().__init__(random=random)
#         self.max_reward = 4
#         self.prev_images = {}

#     def initialize_episode(self, physics):
#         """Sets the state of the environment at the start of each episode."""
#         # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
#         # reset qpos, control and box position
#         with physics.reset_context():
#             physics.named.data.qpos[:16] = START_ARM_POSE
#             np.copyto(physics.data.ctrl, START_ARM_POSE)
#             assert BOX_POSE[0] is not None
#             physics.named.data.qpos[-7:] = BOX_POSE[0]
#             # print(f"{BOX_POSE=}")
#         super().initialize_episode(physics)

#     @staticmethod
#     def get_env_state(physics):
#         env_state = physics.data.qpos.copy()[16:]
#         return env_state

#     def get_reward(self, physics):
#         # return whether left gripper is holding the box
#         all_contact_pairs = []
#         for i_contact in range(physics.data.ncon):
#             id_geom_1 = physics.data.contact[i_contact].geom1
#             id_geom_2 = physics.data.contact[i_contact].geom2
#             name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
#             name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
#             contact_pair = (name_geom_1, name_geom_2)
#             all_contact_pairs.append(contact_pair)

#         touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
#         touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
#         touch_table = ("red_box", "table") in all_contact_pairs

#         reward = 0
#         if touch_right_gripper:
#             reward = 1
#         if touch_right_gripper and not touch_table: # lifted
#             reward = 2
#         if touch_left_gripper: # attempted transfer
#             reward = 3
#         if touch_left_gripper and not touch_table: # successful transfer
#             reward = 4
#         return reward


# class InsertionTask(BimanualViperXTask):
#     def __init__(self, random=None):
#         super().__init__(random=random)
#         self.max_reward = 4

#     def initialize_episode(self, physics):
#         """Sets the state of the environment at the start of each episode."""
#         # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
#         # reset qpos, control and box position
#         with physics.reset_context():
#             physics.named.data.qpos[:16] = START_ARM_POSE
#             np.copyto(physics.data.ctrl, START_ARM_POSE)
#             assert BOX_POSE[0] is not None
#             physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
#             # print(f"{BOX_POSE=}")
#         super().initialize_episode(physics)

#     @staticmethod
#     def get_env_state(physics):
#         env_state = physics.data.qpos.copy()[16:]
#         return env_state

#     def get_reward(self, physics):
#         # return whether peg touches the pin
#         all_contact_pairs = []
#         for i_contact in range(physics.data.ncon):
#             id_geom_1 = physics.data.contact[i_contact].geom1
#             id_geom_2 = physics.data.contact[i_contact].geom2
#             name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
#             name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
#             contact_pair = (name_geom_1, name_geom_2)
#             all_contact_pairs.append(contact_pair)

#         touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
#         touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
#                              ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
#                              ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
#                              ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

#         peg_touch_table = ("red_peg", "table") in all_contact_pairs
#         socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
#                              ("socket-2", "table") in all_contact_pairs or \
#                              ("socket-3", "table") in all_contact_pairs or \
#                              ("socket-4", "table") in all_contact_pairs
#         peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
#                            ("red_peg", "socket-2") in all_contact_pairs or \
#                            ("red_peg", "socket-3") in all_contact_pairs or \
#                            ("red_peg", "socket-4") in all_contact_pairs
#         pin_touched = ("red_peg", "pin") in all_contact_pairs

#         reward = 0
#         if touch_left_gripper and touch_right_gripper: # touch both
#             reward = 1
#         if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
#             reward = 2
#         if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
#             reward = 3
#         if pin_touched: # successful insertion
#             reward = 4
#         return reward


# def get_action(master_bot_left, master_bot_right):
#     action = np.zeros(14)
#     # arm action
#     action[:6] = master_bot_left.dxl.joint_states.position[:6]
#     action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
#     # gripper action
#     left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
#     right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
#     normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
#     normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
#     action[6] = normalized_left_pos
#     action[7+6] = normalized_right_pos
#     return action

# def test_sim_teleop():
#     """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
#     from interbotix_xs_modules.arm import InterbotixManipulatorXS

#     BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

#     # source of data
#     master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                               robot_name=f'master_left', init_node=True)
#     master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
#                                               robot_name=f'master_right', init_node=False)

#     # setup the environment
#     env = make_sim_env('sim_transfer_cube')
#     ts = env.reset()
#     episode = [ts]
#     # setup plotting
#     ax = plt.subplot()
#     plt_img = ax.imshow(ts.observation['images']['angle'])
#     plt.ion()

#     for t in range(1000):
#         action = get_action(master_bot_left, master_bot_right)
#         ts = env.step(action)
#         episode.append(ts)

#         plt_img.set_data(ts.observation['images']['angle'])
#         plt.pause(0.02)


# if __name__ == '__main__':
#     test_sim_teleop()

import numpy as np
import os
import cv2
import collections
import matplotlib.pyplot as plt
import torch
import sys
from RAFT.core.raft import RAFT
os.environ['OMP_NUM_THREADS'] = '1'
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import argparse

# 假设constants.py文件存在，并且定义了所需的常量
# from constants import DT, XML_DIR, START_ARM_POSE
# from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
# from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
# from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
# from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

sys.path.append('RAFT/core')  # raft
import IPython
e = IPython.embed
BOX_POSE = [None]  # to be changed from outside


def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        # xml_path = os.path.join('XML_DIR', f'bimanual_viperx_transfer_cube.xml')
        xml_path = './assets/bimanual_viperx_transfer_cube.xml'
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=0.1,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        # xml_path = os.path.join('XML_DIR', f'bimanual_viperx_insertion.xml')
        xml_path = './assets/bimanual_viperx_insertion.xml'
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=0.1,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        # self.flow = OpticalFlowAttach()
        self.prev_image = {}
        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        self.raft_model = RAFT(args)
        # self.raft_model.load_state_dict(torch.load('raft_weights.pth'))
        self.raft_model = self.raft_model.to('cuda').eval()  # 预训练权重路径

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = 0  # 假设的函数调用
        right_gripper_action = 0  # 假设的函数调用

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)
        # TODO: manipulate the optical flow of the camera
        for cam in ['top', 'left_wrist', 'right_wrist']:
            img = physics.render(height=480, width=640, camera_id=cam)
            self.prev_image[cam] = img.copy()
            # self.flow.prev_images[cam] = img.copy()
            # self.flow.curr_images[cam] = img.copy()

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [0]  # 假设的函数调用
        right_gripper_qpos = [0]  # 假设的函数调用
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [0]  # 假设的函数调用
        right_gripper_qvel = [0]  # 假设的函数调用
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        # obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        for cam in ['top', 'left_wrist', 'right_wrist']:
            obs['images'][cam] = physics.render(height=480, width=640, camera_id=cam)

        # # 使用RAFT计算光流
        # obs['flows'] = {}
        # for cam, cur_img in obs['images'].items():
        #     prev_img = self.prev_image.get(cam, cur_img.copy())

        #     # 将图像转为PyTorch Tensor并归一化
        #     prev_tensor = torch.from_numpy(prev_img).permute(2, 0, 1).float().to('cuda') / 255.0
        #     cur_tensor = torch.from_numpy(cur_img).permute(2, 0, 1).float().to('cuda') / 255.0

        #     # RAFT计算光流
        #     with torch.no_grad():
        #         flow_low, flow_up = self.raft_model(prev_tensor.unsqueeze(0), cur_tensor.unsqueeze(0), iters=20, test_mode=True)
        #         flow = flow_up[0].cpu().numpy()  # [H,W,2]

        #     obs['flows'][cam] = flow
        #     self.prev_image[cam] = cur_img.copy()
        obs['flows'] = {}
        for cam, cur_img in obs['images'].items():
            prev_img = self.prev_image.get(cam, cur_img.copy())

            # 将图像转为PyTorch Tensor并归一化
            prev_tensor = torch.from_numpy(prev_img).permute(2, 0, 1).float().to('cuda') / 255.0
            cur_tensor = torch.from_numpy(cur_img.copy()).permute(2, 0, 1).float().to('cuda') / 255.0
            # cur_tensor = torch.from_numpy(cur_img).permute(2, 0, 1).float().to('cuda') / 255.0

            # RAFT计算光流
            with torch.no_grad():
                _, flow_up = self.raft_model(prev_tensor.unsqueeze(0), 
                                        cur_tensor.unsqueeze(0), 
                                        iters=20, 
                                        test_mode=True)
                
                # 格式兼容性处理
                flow = flow_up[0].cpu().numpy()  # [H,W,2]
                flow = flow.astype(np.float32)   # 确保与OpenCV相同的精度
                
                # 值范围适配（关键修改！）
                flow = flow / 10.0  # 将RAFT的输出缩放至Farneback的量级
                
                # 可选：截断异常值（如果后续模块对大幅值敏感）
                flow = np.clip(flow, -50, 50)  # 限制在±50像素内

            obs['flows'][cam] = flow
            self.prev_image[cam] = cur_img.copy()
        # ## Optical Flow
        # obs['flows'] = {}
        # for cam, cur_img in obs['images'].items():
        #     prev_img = self.prev_images[cam]
        #     prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        #     cur_gray  = cv2.cvtColor(cur_img,  cv2.COLOR_BGR2GRAY)
        #     # Farneback
        #     flow = cv2.calcOpticalFlowFarneback(
        #         prev_gray, cur_gray,
        #         None,
        #         pyr_scale=0.5, levels=3,
        #         winsize=15, iterations=3,
        #         poly_n=5, poly_sigma=1.2, flags=0
        #     )  # flow: (h, w, 2)
        #     obs['flows'][cam] = flow
        #     # Update previous image
        #     self.prev_images[cam] = cur_img.copy()
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.prev_images = {}

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = [0] * 16  # 假设的初始姿势
            np.copyto(physics.data.ctrl, [0] * 16)  # 假设的初始姿势
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = [0] * 16  # 假设的初始姿势
            np.copyto(physics.data.ctrl, [0] * 16)  # 假设的初始姿势
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2:] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = [0] * 6  # 假设的关节位置
    action[7:7 + 6] = [0] * 6  # 假设的关节位置
    # gripper action
    left_gripper_pos = 0  # 假设的关节位置
    right_gripper_pos = 0  # 假设的关节位置
    normalized_left_pos = 0  # 假设的归一化位置
    normalized_right_pos = 0  # 假设的归一化位置
    action[6] = normalized_left_pos
    action[7 + 6] = normalized_right_pos
    return action


def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    # from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    # master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                           robot_name=f'master_left', init_node=True)
    # master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
    #                                           robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['top'])
    plt.ion()

    for t in range(1000):
        action = get_action(None, None)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['top'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()