"""Motion dataloader for loading and sampling motion data from NPZ files.

This module provides a PyTorch-style dataloader with weighted sampling support
and efficient vectorized batch indexing for multi-motion tracking in 
reinforcement learning environments.
"""
from collections.abc import Sequence
import torch
from itertools import accumulate
import bisect
import math
import numpy as np
from typing import Any, Dict, Optional

from mjlab.utils.motion_dataset import Motion_Dataset, Unify_Motion_Dataset

class Motion_Dataloader:
    """Dataloader for sampling motion clips with optional weighted sampling.
    
    Uses efficient concatenation + offset indexing for vectorized batch access.
    All motion sequences are concatenated into single tensors with offset tracking.
    
    Args:
        dataset: Motion_Dataset instance
        device: Device to load tensors on
        
    Example:
        >>> dataset = Motion_Dataset(...)
        >>> dataloader = Motion_Dataloader(dataset)
        >>> 
        >>> # Direct buffer access with global indexing
        >>> motion_ids = torch.tensor([0, 1, 0, 2])
        >>> time_steps = torch.tensor([10, 20, 15, 5])
        >>> global_indices = dataloader.motion_offsets[motion_ids] + time_steps
        >>> joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
        >>> 
        >>> # Uniform sampling
        >>> indices = dataloader.sample(n=10)
        >>> 
        >>> # Weighted sampling
        >>> weights = compute_weights(...)
        >>> indices = dataloader.sample(n=10, weights=weights)
    """
    
    class MotionBuffer:
        """Internal class for storing concatenated motion data tensors.
        
        This class encapsulates all motion data in a single contiguous memory layout
        for efficient GPU-accelerated batch indexing.
        
        Attributes:
            joint_pos: [total_frames, num_joints] - Joint positions
            joint_vel: [total_frames, num_joints] - Joint velocities
            body_pos_w: [total_frames, num_bodies, 3] - Body positions (world frame)
            body_quat_w: [total_frames, num_bodies, 4] - Body quaternions (world frame)
            body_lin_vel_w: [total_frames, num_bodies, 3] - Body linear velocities
            body_ang_vel_w: [total_frames, num_bodies, 3] - Body angular velocities
        """

        def __init__(self, body_indexes: Sequence[int]):
            """Initialize empty motion buffer."""
            self.joint_pos: torch.Tensor | None = None
            self.joint_vel: torch.Tensor | None = None
            self._body_pos_w: torch.Tensor | None = None
            self._body_quat_w: torch.Tensor | None = None
            self._body_lin_vel_w: torch.Tensor | None = None
            self._body_ang_vel_w: torch.Tensor | None = None
            self.body_indexes = body_indexes
        
        @property
        def body_pos_w(self) -> torch.Tensor:
            return self._body_pos_w[:, self.body_indexes]

        @property
        def body_quat_w(self) -> torch.Tensor:
            return self._body_quat_w[:, self.body_indexes]

        @property
        def body_lin_vel_w(self) -> torch.Tensor:
            return self._body_lin_vel_w[:, self.body_indexes]

        @property
        def body_ang_vel_w(self) -> torch.Tensor:
            return self._body_ang_vel_w[:, self.body_indexes]

    
    def __init__(
        self,
        dataset: Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        enable_data_split: bool = False,
    ):
        """Initialize the dataloader with concatenated sequences.
        
        Args:
            dataset: Motion_Dataset instance
            body_indexes: Indices of bodies to extract
            device: Device to load tensors on
            world_size: Total number of distributed processes (default: 1 for single process)
            rank: Current process rank (default: 0)
            enable_data_split: Whether to enable distributed data sharding (default: False)
        """
        self.dataset = dataset
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.enable_data_split = enable_data_split
        
        self._body_indexes = body_indexes
        
        # Initialize motion buffer
        self.motion_buffer = self.MotionBuffer(self._body_indexes)
        
        # Motion metadata (will be populated in _preload_and_concatenate)
        self.motion_lengths: torch.Tensor  # [num_motions_local], length of each motion in this rank
        self.motion_offsets: torch.Tensor  # [num_motions_local], starting index of each motion
        self.motion_fps: torch.Tensor      # [num_motions_local], FPS of each motion
        self.time_step_total: int          # Total number of frames in this rank's buffer
        self.num_motions: int              # Number of motions assigned to this rank
        
        # Distributed data tracking
        self.start_motion_idx: int = 0     # Start motion index in global dataset
        self.end_motion_idx: int = 0       # End motion index in global dataset
        self.global_num_motions: int = len(dataset)  # Total motions in dataset
        
        print(f"[Motion_Dataloader] Loading and concatenating motions for rank {self.rank}/{self.world_size}...")
        
        # Load all motions and concatenate into single tensors
        self._preload_and_concatenate()
        
        print(f"[Motion_Dataloader] Rank {self.rank} initialization complete. Total frames: {self.time_step_total}")
    
    def _preload_and_concatenate(self):
        """Preload all motions and concatenate into single tensors with offset tracking.
        
        This method loads all motion data upfront and concatenates sequences along
        the time dimension. Each motion's starting position is tracked in offsets.
        
        For distributed training: if enable_data_split=True, this will partition motions
        across ranks based on frame count to balance load.
        
        Memory-efficient: No padding, only raw data storage.
        """
        # === Step 1: Load motion metadata for all motions (no GPU transfer yet) ===
        all_motion_lengths = []
        for i in range(self.global_num_motions):
            sample = self.dataset[i]
            all_motion_lengths.append(sample["length"])
        
        # === Step 2: Determine which motions to load for this rank ===
        if self.enable_data_split and self.world_size > 1:
            self._compute_rank_motion_indices(all_motion_lengths)
        else:
            # Single process or data split disabled: load all motions
            self.start_motion_idx = 0
            self.end_motion_idx = self.global_num_motions
        
        self.num_motions = self.end_motion_idx - self.start_motion_idx
        
        # === Step 3: Load and concatenate only this rank's motions ===
        data_lists = {
            'joint_pos': [],
            'joint_vel': [],
            'body_pos_w': [],
            'body_quat_w': [],
            'body_lin_vel_w': [],
            'body_ang_vel_w': [],
        }
        lengths = []
        fps_list = []
        
        # Load motions assigned to this rank
        for i in range(self.start_motion_idx, self.end_motion_idx):
            sample = self.dataset[i]
            motion_data = sample["motion"]
            
            # Append to lists
            for key in data_lists.keys():
                data_lists[key].append(
                    torch.tensor(motion_data[key], dtype=torch.float32, device=self.device)
                )
            
            lengths.append(sample["length"])
            fps_list.append(sample["fps"])
        
        # Concatenate all sequences in this rank's buffer
        self.motion_buffer.joint_pos = torch.cat(data_lists['joint_pos'], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists['joint_vel'], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists['body_pos_w'], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists['body_quat_w'], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists['body_lin_vel_w'], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists['body_ang_vel_w'], dim=0)
        
        # Compute local offsets for this rank (relative to rank's buffer)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat([
            torch.tensor([0], device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1]
        ], dim=0)
        
        # Store FPS for this rank
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        
        # Store total buffer length for this rank
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]
        
        print(f"[Motion_Dataloader] Rank {self.rank} concatenated tensors:")
        print(f"  - Motion indices: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.num_motions}")
        print(f"  - joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  - joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  - body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  - total_frames: {self.time_step_total}")
        print(f"  - motion_lengths range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")
    
    def _compute_rank_motion_indices(self, all_motion_lengths: list[int]):
        """Compute which motion indices are assigned to this rank.
        
        Strategy: Assign complete motions to ranks such that each rank gets roughly
        equal number of frames. Motions are never split across ranks.
        Ensures deterministic, non-overlapping, continuous ranges across all ranks.
        
        Args:
            all_motion_lengths: List of frame counts for all motions in dataset
        """
        total_frames = sum(all_motion_lengths)
        target_frames_per_rank = math.ceil(total_frames / self.world_size) + 1 # for safety, when world_size=1
        
        # cumsum
        cumulative_lengths = list(accumulate(all_motion_lengths))
        cur_rank_tg_start_frames = self.rank * target_frames_per_rank
        cur_rank_tg_end_frames = (self.rank + 1) * target_frames_per_rank
        
        # find motion index 
        start_motion_idx = bisect.bisect_left(cumulative_lengths.copy(), cur_rank_tg_start_frames)
        end_motion_idx = bisect.bisect_right(cumulative_lengths.copy(), cur_rank_tg_end_frames)
        self.start_motion_idx = start_motion_idx
        self.end_motion_idx = end_motion_idx
        
        rank_total_frames = sum(all_motion_lengths[i] for i in range(self.start_motion_idx, self.end_motion_idx))
        
        print(f"[Motion_Dataloader] Rank {self.rank}/{self.world_size} motion assignment:")
        print(f"  - Motion range: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.end_motion_idx - self.start_motion_idx}")
        print(f"  - Target frames per rank: {target_frames_per_rank}")
        print(f"  - Assigned frames: {rank_total_frames}")

    def get_motion_length(self, motion_id: int) -> int:
        """Get length of a specific motion."""
        return self.motion_lengths[motion_id].item()
    
    def get_motion_fps(self, motion_id: int) -> float:
        """Get FPS of a specific motion."""
        return self.motion_fps[motion_id].item()
    
    def sample(self, n: int, weights: torch.Tensor | list | None = None) -> torch.Tensor:
        """Sample n motion indices with optional weights.
        
        Args:
            n: Number of motion clips to sample
            weights: Optional [num_motions] tensor or list of sampling weights.
                    If None, uniform sampling is used.
                    Weights will be normalized internally.
        
        Returns:
            motion_indices: Tensor[n], sampled motion indices in dataset
            
        Example:
            # Uniform sampling
            indices = dataloader.sample(10)
            
            # Weighted sampling based on quantity
            weights = [0.85 if q==1 else 0.10 if q==2 else 0.05 
                      for q in dataset.quantities]
            indices = dataloader.sample(10, weights=weights)
            
            # Custom adaptive sampling
            weights = curriculum_weights * difficulty_scores * diversity_penalty
            indices = dataloader.sample(10, weights=weights)
        """
        if weights is None:
            # Uniform sampling
            weights = torch.ones(self.num_motions, device=self.device)
        else:
            # Convert to tensor if needed
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                weights = weights.to(self.device)
            
            # Validate shape
            if weights.shape[0] != self.num_motions:
                raise ValueError(
                    f"Weights shape mismatch: expected [{self.num_motions}], got {weights.shape}"
                )
        
        # Ensure positive weights
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        motion_indices = torch.multinomial(weights, n, replacement=True)
        
        return motion_indices
    

class Unify_Motion_Dataloader(Motion_Dataloader):
    """Extended dataloader for paired robot and SMPL-X motion data.
    
    Inherits from Motion_Dataloader and adds support for:
    - smplx_pose_body: [total_frames, 126] - SMPL-X body pose flattened
    - robot_keypoints_trans: [total_frames, 15] - Robot keypoints translation flattened
    - robot_keypoints_rot: [total_frames, 30] - Robot keypoints rotation in 6D flattened
    
    Note: Extended keys are already flattened in Unify_Motion_Dataset.__getitem__()
    
    Args:
        dataset: Unify_Motion_Dataset instance (paired robot + SMPL-X)
        body_indexes: Indices for filtering robot body data
        device: Device to load tensors on
        
    Example:
        >>> dataset = Unify_Motion_Dataset(robot_map, smplx_map, robot_name="adam_sp")
        >>> dataloader = Unify_Motion_Dataloader(dataset, body_indexes=[0,1,2,...], device="cuda")
        >>> 
        >>> # Use inherited sampling method
        >>> indices = dataloader.sample(n=32)
        >>> time_steps = torch.tensor([10, 20, 15, ...], device="cuda")
        >>> global_indices = dataloader.motion_offsets[indices] + time_steps
        >>> 
        >>> # Access all data including extended keys
        >>> joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
        >>> smplx_data = dataloader.motion_buffer.smplx_pose_body[global_indices]
        >>> keypoint_trans = dataloader.motion_buffer.robot_keypoints_trans[global_indices]
    """
    
    class UnifyMotionBuffer(Motion_Dataloader.MotionBuffer):
        """Extended MotionBuffer with SMPL-X and extended data support.
        
        Inherits all robot motion buffers from parent and adds:
        - smplx_pose_body: SMPL-X body pose data
        - robot_keypoints_trans: Robot keypoints translation
        - robot_keypoints_rot: Robot keypoints rotation in 6D
        """
        
        def __init__(self, body_indexes: Sequence[int]):
            super().__init__(body_indexes)
            self._smplx_pose_body = None
            self._robot_keypoints_trans = None
            self._robot_keypoints_rot = None
        
        @property
        def smplx_pose_body(self) -> torch.Tensor | None:
            """SMPL-X body pose in 6D representation."""
            return self._smplx_pose_body
        
        @property
        def robot_keypoints_trans(self) -> torch.Tensor | None:
            """Robot keypoints SE3 translation data."""
            return self._robot_keypoints_trans
        
        @property
        def robot_keypoints_rot(self) -> torch.Tensor | None:
            """Robot keypoints SE3 rotation in 6D representation."""
            return self._robot_keypoints_rot
    
    
    def __init__(
        self,
        dataset: Unify_Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        enable_data_split: bool = False,
    ):
        """Initialize Unify_Motion_Dataloader.
        
        Args:
            dataset: Unify_Motion_Dataset instance with paired data
            body_indexes: Sequence of body indices for filtering
            device: Device to load tensors on
            world_size: Total number of distributed processes (default: 1 for single process)
            rank: Current process rank (default: 0)
            enable_data_split: Whether to enable distributed data sharding (default: False)
        """
        self.dataset = dataset
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.enable_data_split = enable_data_split
        
        self._body_indexes = body_indexes
        
        # Use extended buffer class
        self.motion_buffer = self.UnifyMotionBuffer(self._body_indexes)
        
        # Initialize metadata (will be populated in _preload_and_concatenate)
        self.motion_lengths: torch.Tensor
        self.motion_offsets: torch.Tensor
        self.motion_fps: torch.Tensor
        self.time_step_total: int
        self.num_motions : int
        
        # Distributed data tracking
        self.start_motion_idx: int = 0     # Start motion index in global dataset
        self.end_motion_idx: int = 0       # End motion index in global dataset
        self.global_num_motions: int = len(dataset)  # Total motions in dataset
        
        print(f"[Unify_Motion_Dataloader] Loading and concatenating motions for rank {self.rank}/{self.world_size}...")
        
        # Load all motions and concatenate
        self._preload_and_concatenate()
        
        print(f"[Unify_Motion_Dataloader] Rank {self.rank} initialization complete. Total frames: {self.time_step_total}")
    
    
    def _preload_and_concatenate(self) -> None:
        """Preload all paired motions and concatenate with extended data handling.
        
        Extends parent method to also handle SMPL-X and extended robot data.
        For distributed training: if enable_data_split=True, this will partition motions
        across ranks based on frame count to balance load.
        """
        # === Step 1: Load motion metadata for all motions (no GPU transfer yet) ===
        all_motion_lengths = []
        for i in range(self.global_num_motions):
            sample = self.dataset[i]
            all_motion_lengths.append(sample["length"])
        
        # === Step 2: Determine which motions to load for this rank ===
        if self.enable_data_split and self.world_size > 1:
            self._compute_rank_motion_indices(all_motion_lengths)
        else:
            # Single process or data split disabled: load all motions
            self.start_motion_idx = 0
            self.end_motion_idx = self.global_num_motions
        
        self.num_motions = self.end_motion_idx - self.start_motion_idx
        
        # === Step 3: Load and concatenate only this rank's motions ===
        # Base data lists (inherited from parent)
        data_lists = {
            'joint_pos': [],
            'joint_vel': [],
            'body_pos_w': [],
            'body_quat_w': [],
            'body_lin_vel_w': [],
            'body_ang_vel_w': [],
        }
        
        # Extended data lists for SMPL-X and robot keypoints
        # Note: These are already flattened by Unify_Motion_Dataset
        extended_lists = {
            'smplx_pose_body': [],
            'robot_keypoints_trans': [],
            'robot_keypoints_rot': [],
        }
        
        lengths = []
        fps_list = []
        
        # Load motions assigned to this rank
        for i in range(self.start_motion_idx, self.end_motion_idx):
            item = self.dataset[i]
            robot_motion = item["motion"]
            robot_len = item["length"]
            robot_item = item
            
            # Load base data (reuse from Motion_Dataset interface)
            for key in data_lists.keys():
                data_lists[key].append(
                    torch.tensor(robot_motion[key], dtype=torch.float32, device=self.device)
                )
            
            # Load extended data (must exist from extend_datasets.py preprocessing)
            # Note: Data is already flattened by Unify_Motion_Dataset.__getitem__()
            for ext_key in extended_lists.keys():
                extended_lists[ext_key].append(
                    torch.tensor(robot_motion[ext_key], dtype=torch.float32, device=self.device)
                )
            
            lengths.append(robot_len)
            fps_list.append(robot_item["fps"])
        
        # Concatenate base data (robot motion)
        self.motion_buffer.joint_pos = torch.cat(data_lists['joint_pos'], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists['joint_vel'], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists['body_pos_w'], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists['body_quat_w'], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists['body_lin_vel_w'], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists['body_ang_vel_w'], dim=0)
        
        # Concatenate extended data (must exist from extend_datasets.py preprocessing)
        self.motion_buffer._smplx_pose_body = torch.cat(extended_lists['smplx_pose_body'], dim=0)
        self.motion_buffer._robot_keypoints_trans = torch.cat(extended_lists['robot_keypoints_trans'], dim=0)
        self.motion_buffer._robot_keypoints_rot = torch.cat(extended_lists['robot_keypoints_rot'], dim=0)
        
        # Compute motion metadata
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long, device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1]
        ], dim=0)
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]
        
        # Print buffer info
        print(f"[Unify_Motion_Dataloader] Rank {self.rank} concatenated tensors:")
        print(f"  - Motion indices: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.num_motions}")
        print(f"  joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  smplx_pose_body: {self.motion_buffer._smplx_pose_body.shape}")
        print(f"  robot_keypoints_trans: {self.motion_buffer._robot_keypoints_trans.shape}")
        print(f"  robot_keypoints_rot: {self.motion_buffer._robot_keypoints_rot.shape}")
        print(f"  total_frames: {self.time_step_total}")
        print(f"  motion_lengths: {self.motion_lengths.shape}, range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")
        print(f"  motion_offsets: {self.motion_offsets.shape}")
