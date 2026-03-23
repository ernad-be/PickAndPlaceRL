from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 135.0,
    "elevation": -25.0,
    "lookat": np.array([0.4, 0.0, 0.2]),
}

# Scene XML path relative to this file
_XML_PATH = os.path.join(os.path.dirname(__file__), "franka_emika_panda", "pick_and_place_scene.xml")

# Robot constants
_N_ARM_JOINTS = 7
_N_FINGER_JOINTS = 2
_N_ROBOT_JOINTS = _N_ARM_JOINTS + _N_FINGER_JOINTS  # 9

# Observation dimensions (flat mode)
# arm_qpos(7) + arm_qvel(7) + finger_qpos(2) + finger_qvel(2) + grip_pos(3) + grip_velp(3)
# + object_pos(3) + object_quat(4) + object_vel(6) + rel_obj_grip(3) + target_pos(3)
# + rel_obj_target(3) + contact_flags(2)
_OBS_DIM = 48

# Workspace limits for randomization
_OBJECT_X_RANGE = (0.35, 0.65)
_OBJECT_Y_RANGE = (-0.3, 0.3)
_OBJECT_Z_INIT = 0.02  # On table surface

_MIN_OBJECT_TARGET_DIST = 0.05  # Minimum distance between object spawn and target

# Robot home configuration (matches keyframe "home" in panda.xml)
# joint1..7 + finger_left + finger_right
_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])

# Home ctrl: 7 arm joint targets + gripper fully open (255)
_HOME_CTRL = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0])

# Fixed target position [x, y, z] in world frame
# Centred in workspace, raised 20 cm above the table
_FIXED_TARGET_POS = np.array([0.5, 0.0, 0.2])

# Per-episode target randomization around the fixed centre (XY jitter only)
_TARGET_X_RANGE = (_FIXED_TARGET_POS[0] - 0.1, _FIXED_TARGET_POS[0] + 0.1)
_TARGET_Y_RANGE = (_FIXED_TARGET_POS[1] - 0.1, _FIXED_TARGET_POS[1] + 0.1)
_TARGET_Z = _FIXED_TARGET_POS[2]  # Fixed height — add Z jitter once grasping is learned

# Identity quaternion [w, x, y, z] for object initial orientation
_OBJECT_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Contact force thresholds (Newtons, normal component)
_TOUCH_FORCE_THRESHOLD = 0.25  # Minimum normal force to count as a touch
_GRASP_FORCE_THRESHOLD = 0.5  # Minimum normal force per finger to count as a grasp
_MIN_GRASP_GRIPPER_WIDTH = 0.015  # Minimum finger opening (m) to count as a valid grasp
_GRASP_OPPOSING_FORCE_COSINE_MAX = -0.15  # Require per-finger force directions to be approximately opposite

# Cartesian velocity control
_MAX_CARTESIAN_STEP = 0.05  # metres per control step (~1.25 m/s at 25 Hz)
_DLS_DAMPING = 1e-4  # Damped least-squares regularisation for Jacobian pseudoinverse
_MAX_JOINT_DELTA = 0.5  # max joint-angle change (rad) per control step

# Reward penalties
_FLOOR_CONTACT_PENALTY = -1.0  # Per-step penalty when any robot link touches the floor
_ACTION_PENALTY_COEF = 0.01  # Small L2 penalty on actions to encourage smooth motion


class PandaPickAndPlaceEnv(MujocoEnv):
    """Franka Emika Panda pick-and-place environment.

    A 7-DOF robot arm with a parallel-jaw gripper must pick up a small box
    from the table and place it at a fixed target location.

    Observation space (flat, 48-dim):
        - arm joint positions (7)
        - arm joint velocities (7)
        - finger joint positions (2)
        - finger joint velocities (2)
        - gripper centre position in world frame (3)
        - gripper centre linear velocity in world frame (3)
        - object position (3)
        - object orientation quaternion (4)
        - object velocity (6: 3 linear + 3 angular)
        - relative vector: object - gripper (3)
        - target position (3)
        - relative vector: object - target (3)
        - contact flags: touched, grasped (2)

    Action space (4-dim, continuous [-1, 1]):
        - dx, dy, dz: end-effector Cartesian velocity (scaled by _MAX_CARTESIAN_STEP)
        - gripper: -1 = fully closed, +1 = fully open

    Args:
        render_mode: Gymnasium render mode ("human", "rgb_array", or None).
        reward_type: "dense" for shaped reward, "sparse" for binary reward.
        distance_threshold: Distance below which the task is considered solved.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        reward_type: str = "dense",
        distance_threshold: float = 0.05,
    ) -> None:
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        # Build observation space
        obs_space = Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float64)

        super().__init__(
            model_path=_XML_PATH,
            frame_skip=20,  # 0.002s timestep * 20 = 0.04s per step (25 Hz control)
            observation_space=obs_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        # Cache body IDs for efficient lookup
        self._object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self._grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Cache geom ID sets for contact-based grasp detection.
        # Use only fingertip pad geoms (the 5 box pads per finger), not visual meshes.
        left_finger_geom_ids = set(
            range(
                self.model.body_geomadr[self._left_finger_body_id],
                self.model.body_geomadr[self._left_finger_body_id] + self.model.body_geomnum[self._left_finger_body_id],
            )
        )
        right_finger_geom_ids = set(
            range(
                self.model.body_geomadr[self._right_finger_body_id],
                self.model.body_geomadr[self._right_finger_body_id]
                + self.model.body_geomnum[self._right_finger_body_id],
            )
        )
        self._left_finger_pad_geom_ids = frozenset(
            gid for gid in left_finger_geom_ids if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        )
        self._right_finger_pad_geom_ids = frozenset(
            gid for gid in right_finger_geom_ids if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        )
        if not self._left_finger_pad_geom_ids:
            self._left_finger_pad_geom_ids = frozenset(left_finger_geom_ids)
        if not self._right_finger_pad_geom_ids:
            self._right_finger_pad_geom_ids = frozenset(right_finger_geom_ids)
        self._object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object")

        # Floor geom + robot body IDs for floor-contact penalty
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # All robot link bodies (exclude link0 = fixed base on the ground)
        self._robot_link_body_ids: set[int] = set()
        for name in ("link1", "link2", "link3", "link4", "link5", "link6", "link7",
                     "hand", "left_finger", "right_finger"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._robot_link_body_ids.add(bid)
        # Build set of collision geom IDs belonging to those bodies
        self._robot_collision_geom_ids: frozenset[int] = frozenset(
            gid
            for bid in self._robot_link_body_ids
            for gid in range(
                self.model.body_geomadr[bid],
                self.model.body_geomadr[bid] + self.model.body_geomnum[bid],
            )
            if self.model.geom_contype[gid] > 0 or self.model.geom_conaffinity[gid] > 0
        )

        # Object freejoint starts after robot joints in qpos/qvel
        self._object_qpos_start = _N_ROBOT_JOINTS  # index 9
        self._object_qvel_start = _N_ROBOT_JOINTS  # index 9 (qvel for freejoint is 6-dim)

        # --- Cartesian velocity control ---
        # Override the actuator-derived action space with a 4-D box.
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Pre-allocate Jacobian buffer (3 × nv)
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        # Pre-allocate observation buffer (avoids np.concatenate per step)
        self._obs_buf = np.zeros(_OBS_DIM, dtype=np.float64)
        # Pre-allocate wrench buffer for contact force queries
        self._wrench_buf = np.zeros(6, dtype=np.float64)
        # Cache arm joint limits for clipping
        self._arm_jnt_low = self.model.jnt_range[:_N_ARM_JOINTS, 0].copy()
        self._arm_jnt_high = self.model.jnt_range[:_N_ARM_JOINTS, 1].copy()

    # ------------------------------------------------------------------
    # Contact detection (single-pass)
    # ------------------------------------------------------------------

    def _scan_all_contacts(self) -> tuple[bool, bool, float, float, bool]:
        """Single-pass scan of all contacts for grasp detection AND floor penalty.

        Returns:
            touched, grasped, left_force, right_force, robot_on_floor
        """
        left_force = 0.0
        right_force = 0.0
        left_force_vec = np.zeros(3, dtype=np.float64)
        right_force_vec = np.zeros(3, dtype=np.float64)
        wrench = self._wrench_buf
        robot_on_floor = False

        obj_geom = self._object_geom_id
        floor_geom = self._floor_geom_id
        robot_geoms = self._robot_collision_geom_ids
        left_pads = self._left_finger_pad_geom_ids
        right_pads = self._right_finger_pad_geom_ids

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            # --- Floor-contact check ---
            if not robot_on_floor:
                if (g1 == floor_geom and g2 in robot_geoms) or \
                   (g2 == floor_geom and g1 in robot_geoms):
                    robot_on_floor = True

            # --- Object-finger contact ---
            if g1 == obj_geom:
                other_geom = g2
            elif g2 == obj_geom:
                other_geom = g1
            else:
                continue

            is_left = other_geom in left_pads
            is_right = other_geom in right_pads
            if not (is_left or is_right):
                continue

            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            normal_force = abs(wrench[0])

            contact_normal = np.array(contact.frame[:3], dtype=np.float64)
            inward = contact_normal if g2 == obj_geom else -contact_normal

            if is_left:
                left_force += normal_force
                left_force_vec += normal_force * inward
            else:
                right_force += normal_force
                right_force_vec += normal_force * inward

        touched = (left_force >= _TOUCH_FORCE_THRESHOLD) or (right_force >= _TOUCH_FORCE_THRESHOLD)

        left_norm = float(np.linalg.norm(left_force_vec))
        right_norm = float(np.linalg.norm(right_force_vec))
        forces_opposing = False
        if left_norm > 1e-8 and right_norm > 1e-8:
            cosine = float(np.dot(left_force_vec, right_force_vec) / (left_norm * right_norm))
            forces_opposing = cosine <= _GRASP_OPPOSING_FORCE_COSINE_MAX

        gripper_width = float(self.data.qpos[_N_ARM_JOINTS] + self.data.qpos[_N_ARM_JOINTS + 1])
        grasped = (
            (left_force >= _GRASP_FORCE_THRESHOLD)
            and (right_force >= _GRASP_FORCE_THRESHOLD)
            and forces_opposing
            and (gripper_width >= _MIN_GRASP_GRIPPER_WIDTH)
        )
        return touched, grasped, left_force, right_force, robot_on_floor

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(
        self,
        *,
        touched: bool | None = None,
        grasped: bool | None = None,
        grip_pos: np.ndarray | None = None,
        object_pos: np.ndarray | None = None,
        target_pos: np.ndarray | None = None,
        grip_velp: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a 48-dim flat observation vector into a pre-allocated buffer.

        Accepts optional cached values to avoid redundant lookups when
        called from ``step()``.
        """
        buf = self._obs_buf
        d = self.data

        # Joint state (no .copy() needed — written directly into buf)
        buf[0:7] = d.qpos[:_N_ARM_JOINTS]
        buf[7:14] = d.qvel[:_N_ARM_JOINTS]
        buf[14:16] = d.qpos[_N_ARM_JOINTS:_N_ROBOT_JOINTS]
        buf[16:18] = d.qvel[_N_ARM_JOINTS:_N_ROBOT_JOINTS]

        # Gripper position
        if grip_pos is None:
            grip_pos = d.site_xpos[self._grip_site_id]
        buf[18:21] = grip_pos

        # Gripper linear velocity (reuse Jacobian if needed)
        if grip_velp is None:
            self._jacp[:] = 0.0
            mujoco.mj_jacSite(self.model, d, self._jacp, None, self._grip_site_id)
            grip_velp = self._jacp @ d.qvel
        buf[21:24] = grip_velp

        # Object state
        if object_pos is None:
            object_pos = d.xpos[self._object_body_id]
        buf[24:27] = object_pos

        oqs = self._object_qpos_start
        buf[27:31] = d.qpos[oqs + 3 : oqs + 7]

        ovs = self._object_qvel_start
        buf[31:37] = d.qvel[ovs : ovs + 6]

        # Relative vectors
        buf[37:40] = object_pos - grip_pos

        if target_pos is None:
            target_pos = d.site_xpos[self._target_site_id]
        buf[40:43] = target_pos
        buf[43:46] = object_pos - target_pos

        # Contact flags
        if touched is None or grasped is None:
            t, g, _, _, _ = self._scan_all_contacts()
            touched = t if touched is None else touched
            grasped = g if grasped is None else grasped
        buf[46] = 1.0 if touched else 0.0
        buf[47] = 1.0 if grasped else 0.0

        return buf.copy()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self) -> np.ndarray:
        """Reset robot to home pose, randomize target slightly, and randomize object position.

        - Robot joints are set to ``_HOME_QPOS`` (arm + fingers fully open).
        - Target is sampled near ``_FIXED_TARGET_POS`` every episode.
        - Object is placed at a random (x, y) on the table, at least
          ``_MIN_OBJECT_TARGET_DIST`` away from the target.
        - All velocities are zeroed.
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Robot home pose (arm joints + fingers fully open)
        qpos[:_N_ROBOT_JOINTS] = _HOME_QPOS

        # Slightly randomize target around the workspace centre
        target_pos = np.array(
            [
                self.np_random.uniform(*_TARGET_X_RANGE),
                self.np_random.uniform(*_TARGET_Y_RANGE),
                _TARGET_Z,
            ]
        )

        # Randomize object position; retry until far enough from the target
        for _ in range(100):
            object_x = self.np_random.uniform(*_OBJECT_X_RANGE)
            object_y = self.np_random.uniform(*_OBJECT_Y_RANGE)
            object_pos = np.array([object_x, object_y, _OBJECT_Z_INIT])
            # Check 2D (XY) distance so it doesn't spawn directly under the target
            xy_dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
            if xy_dist >= _MIN_OBJECT_TARGET_DIST:
                break

        # Object freejoint: position + identity quaternion (single write)
        qpos[self._object_qpos_start : self._object_qpos_start + 7] = np.concatenate([object_pos, _OBJECT_INIT_QUAT])

        # Zero all velocities
        qvel[:] = 0.0

        # Set ctrl and mocap BEFORE set_state so its internal
        # mj_forward propagates correct values in one pass.
        self.data.ctrl[:] = _HOME_CTRL
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = target_pos

        self.set_state(qpos, qvel)

        # Reset Cartesian controller state
        self._last_arm_target = _HOME_QPOS[:_N_ARM_JOINTS].copy()

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step & reward
    # ------------------------------------------------------------------

    # _check_robot_floor_contact merged into _scan_all_contacts

    def _compute_dense_reward(
        self,
        dist_grip_to_obj: float,
        dist_obj_to_target: float,
        object_pos_z: float,
        object_touched: bool,
        object_grasped: bool,
        is_success: bool,
        robot_on_floor: bool,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Phased dense reward for pick-and-place.

        Returns:
            A ``(total_reward, components_dict)`` tuple.
        """
        total = 0.0

        # ---- Phase 1: REACH (always active) ----
        reach_reward = 0.5 * (1.0 - np.tanh(5.0 * dist_grip_to_obj))
        total += reach_reward

        # ---- Phase 2: GRASP (bonus for touching / grasping the object) ----
        if object_grasped:
            grasp_reward = 0.5
        elif object_touched:
            grasp_reward = 0.1
        else:
            grasp_reward = 0.0
        total += grasp_reward

        # ---- Phase 3: LIFT (reward for raising object above table, only when grasped) ----
        lift_reward = 0.0
        if object_grasped:
            lift_reward = 1.0 * np.tanh(5.0 * max(0.0, object_pos_z - _OBJECT_Z_INIT))
        total += lift_reward

        # ---- Phase 4: PLACE (reward for moving object towards target) ----
        place_reward = 2.0 * (1.0 - np.tanh(5.0 * dist_obj_to_target))
        total += place_reward

        # ---- Success bonus ----
        success_reward = 200.0 if is_success else 0.0
        total += success_reward

        # ---- Penalties ----
        floor_penalty = _FLOOR_CONTACT_PENALTY if robot_on_floor else 0.0
        total += floor_penalty

        action_penalty = -_ACTION_PENALTY_COEF * float(np.sum(action[:3] ** 2))
        total += action_penalty

        components = {
            "reward/reach": reach_reward,
            "reward/grasp": grasp_reward,
            "reward/lift": lift_reward,
            "reward/place": place_reward,
            "reward/success": success_reward,
            "reward/floor_penalty": floor_penalty,
            "reward/action_penalty": action_penalty,
        }
        return total, components

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: 4-dim array [dx, dy, dz, gripper] in [-1, 1].
                    dx/dy/dz are Cartesian velocity commands for the
                    end-effector, scaled by ``_MAX_CARTESIAN_STEP``.
                    gripper: -1 = closed, +1 = open.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # --- Convert Cartesian action to joint-position ctrl ---
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ee_vel = action[:3] * _MAX_CARTESIAN_STEP

        # Compute positional Jacobian of grip site w.r.t. all DoFs
        self._jacp[:] = 0.0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self._grip_site_id)
        J_arm = self._jacp[:, :_N_ARM_JOINTS]  # (3, 7) – only arm joints

        # Damped least-squares pseudoinverse for numerical stability
        JJT = J_arm @ J_arm.T + _DLS_DAMPING * np.eye(3)
        dq = J_arm.T @ np.linalg.solve(JJT, ee_vel)

        # Clamp per-joint delta
        dq = np.clip(dq, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)

        # Integrate from current joint positions
        current_arm_q = self.data.qpos[:_N_ARM_JOINTS].copy()
        new_arm_target = np.clip(
            current_arm_q + dq,
            self._arm_jnt_low,
            self._arm_jnt_high,
        )
        self._last_arm_target = new_arm_target

        # Gripper: map [-1, 1] → [0, 255] (actuator8 ctrlrange)
        gripper_ctrl = float((action[3] + 1.0) / 2.0 * 255.0)

        # Build 8-D ctrl vector (7 arm joint targets + 1 gripper)
        ctrl = np.concatenate([new_arm_target, [gripper_ctrl]])

        # Simulate
        self.do_simulation(ctrl, self.frame_skip)

        # --- Post-simulation: compute everything once ---
        d = self.data

        # Single-pass contact scan (grasp + floor)
        object_touched, object_grasped, left_force, right_force, robot_on_floor = (
            self._scan_all_contacts()
        )

        # Positions (read once, share with obs)
        grip_pos = d.site_xpos[self._grip_site_id]
        object_pos = d.xpos[self._object_body_id]
        target_pos = d.site_xpos[self._target_site_id]

        # Jacobian → grip velocity (computed once, shared with obs)
        self._jacp[:] = 0.0
        mujoco.mj_jacSite(self.model, d, self._jacp, None, self._grip_site_id)
        grip_velp = self._jacp @ d.qvel

        # Observation (all cached values passed in — no redundant lookups)
        obs = self._get_obs(
            touched=object_touched,
            grasped=object_grasped,
            grip_pos=grip_pos,
            object_pos=object_pos,
            target_pos=target_pos,
            grip_velp=grip_velp,
        )

        # Distances
        dist_grip_to_obj = float(np.linalg.norm(grip_pos - object_pos))
        dist_obj_to_target = float(np.linalg.norm(object_pos - target_pos))

        # Success
        is_success = dist_obj_to_target < self.distance_threshold and object_grasped

        # Reward
        reward_components: dict[str, float] = {}
        if self.reward_type == "sparse":
            reward = 0.0 if is_success else -1.0
        else:
            reward, reward_components = self._compute_dense_reward(
                dist_grip_to_obj,
                dist_obj_to_target,
                float(object_pos[2]),
                object_touched,
                object_grasped,
                is_success,
                robot_on_floor,
                action,
            )

        # Terminate early on success or if the object fell off the table.
        object_fell = float(object_pos[2]) < -0.1
        terminated = bool(is_success) or object_fell
        truncated = False

        info: dict[str, Any] = {
            "is_success": is_success,
            "dist_grip_to_obj": dist_grip_to_obj,
            "dist_obj_to_target": dist_obj_to_target,
            "object_touched": object_touched,
            "object_grasped": object_grasped,
            "left_grip_force": left_force,
            "right_grip_force": right_force,
            "object_height": float(object_pos[2]),
            **reward_components,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
