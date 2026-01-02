"""
CMU Motion Capture Nodes for ComfyUI

Provides motion capture playback from the CMU Motion Database.
Supports locomotion with proper drift, sequence chaining, and multi-character pairs.
"""

import os
import json
import numpy as np
import torch
import random
import tarfile
import urllib.request
from pathlib import Path


# HuggingFace URL for CMU data
CMU_HF_URL = "https://huggingface.co/datasets/ckinpdx/CMUMDB/resolve/main/cmu_npy.tar.gz"


def download_cmu_library(destination, max_retries=3):
    """Download and extract CMU motion library from HuggingFace."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    
    index_path = destination / "index.json"
    if index_path.exists():
        return True
    
    tar_path = destination / "cmu_npy.tar.gz"
    
    for attempt in range(max_retries):
        try:
            print(f"[CMU] Downloading motion library (attempt {attempt + 1})...")
            urllib.request.urlretrieve(CMU_HF_URL, tar_path)
            
            # Verify file size
            if tar_path.stat().st_size < 600_000_000:  # ~660MB expected
                print(f"[CMU] Download incomplete, retrying...")
                tar_path.unlink()
                continue
            
            print(f"[CMU] Extracting...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(destination)
            
            tar_path.unlink()
            
            if index_path.exists():
                return True
            else:
                print(f"[CMU] Extraction failed, index.json not found")
                
        except Exception as e:
            print(f"[CMU] Download error: {e}")
            if tar_path.exists():
                tar_path.unlink()
    
    return False


class SCAILCMULibraryLoader:
    """Load CMU Motion Capture library."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["huggingface", "local"],),
            },
            "optional": {
                "local_path": ("STRING", {"default": "", "tooltip": "Path to local CMU npy folder with index.json"}),
            }
        }
    
    RETURN_TYPES = ("CMU_LIBRARY",)
    RETURN_NAMES = ("library",)
    FUNCTION = "load"
    CATEGORY = "SCAIL/Motion"
    
    def load(self, source, local_path=""):
        if source == "local" and local_path:
            base_path = local_path
        else:
            # Get ComfyUI models path
            comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_path = os.path.join(comfy_path, "models", "cmu_motion")
            
            # Download if needed
            if not os.path.exists(os.path.join(base_path, "index.json")):
                success = download_cmu_library(base_path)
                if not success:
                    raise RuntimeError("Failed to download CMU motion library")
        
        # Load index
        index_path = os.path.join(base_path, "index.json")
        if not os.path.exists(index_path):
            raise RuntimeError(f"index.json not found at {base_path}")
            
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        library = {
            'base_path': base_path,
            'index': index,
            'motions': index['motions'],
            'categories': index['categories']
        }
        
        total = len(index['motions'])
        cats = len(index['categories'])
        print(f"[CMU] Loaded library: {total} motions in {cats} categories")
        
        return (library,)


class SCAILCMUMotion:
    """
    Single CMU motion node. Chain multiple for sequences.
    Preserves locomotion drift and handles transitions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": ("CMU_LIBRARY",),
                "category": ([
                    "walk", "run", "jump", "dance", "fight", "sports",
                    "climb", "interact", "sit", "crouch", "gesture",
                    "swim", "getup", "fall", "balance", "misc"
                ],),
                "selection_mode": (["random", "specific"],),
                "motion_id": ("STRING", {"default": "", "tooltip": "Motion ID when selection_mode is 'specific' (e.g., '02_01')"}),
                "transition_frames": ("INT", {"default": 12, "min": 0, "max": 60}),
                "loop_mode": (["none", "loop", "ping_pong"],),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
            "optional": {
                "scail_pose": ("SCAIL_POSE",),
                "pose_sequence_in": ("SCAIL_POSE_SEQUENCE",),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("pose_sequence", "frame_count_out", "motion_info")
    FUNCTION = "generate"
    CATEGORY = "SCAIL/Motion"
    
    def generate(self, library, category, selection_mode, motion_id, transition_frames, loop_mode,
                 target_fps, scail_pose=None, pose_sequence_in=None):
        
        base_path = library['base_path']
        index = library['index']
        source_fps = index.get('fps', 120)
        time_scale = source_fps / target_fps
        
        # Select motion
        if selection_mode == "random" or not motion_id:
            available = library['categories'].get(category, [])
            if not available:
                available = list(index['motions'].keys())
            motion_id = random.choice(available)
        elif motion_id not in index['motions']:
            print(f"[CMU] Motion '{motion_id}' not found, selecting random from {category}")
            available = library['categories'].get(category, [])
            if not available:
                available = list(index['motions'].keys())
            motion_id = random.choice(available)
        
        motion_info = index['motions'][motion_id]
        motion_path = os.path.join(base_path, motion_info['file'])
        
        # Load motion data
        motion_data = np.load(motion_path)
        motion_frames = len(motion_data)
        
        # Calculate output frame count from motion length and fps scaling
        frame_count = int(motion_frames / time_scale)
        
        motion_desc = f"{motion_id}: {motion_info['description']}"
        print(f"[CMU] Playing: {motion_desc[:70]} ({motion_frames} @ {source_fps}fps -> {frame_count} @ {target_fps}fps)")
        
        # Get frame count from input sequence
        if pose_sequence_in is not None:
            start_frame = pose_sequence_in['frame_count']
            input_poses = pose_sequence_in['poses']
            last_position_x = pose_sequence_in.get('last_position_x', None)
            last_pose = input_poses[-1] if len(input_poses) > 0 else None
        else:
            start_frame = 0
            input_poses = []
            last_position_x = None
            last_pose = None
        
        # Get reference data for scaling/positioning
        if scail_pose is not None:
            ref_joints = scail_pose['joints'].numpy()
            char_count = scail_pose.get('char_count', 1)
        else:
            ref_joints = None
            char_count = 1
        
        # Build reference data per character
# Get reference data for scaling/positioning
        ref_data = []
        
        if scail_pose is not None:
            ref_joints = scail_pose['joints'].numpy()
            char_count = scail_pose.get('char_count', 1)
            
            # Build reference data per character
            for i in range(char_count):
                start_idx = i * 18
                end_idx = start_idx + 18
                if end_idx <= len(ref_joints):
                    char_joints = ref_joints[start_idx:end_idx]
                    
                    # Helper to check if joint is valid (not zero/origin)
                    def is_valid(joint_idx):
                        j = char_joints[joint_idx]
                        return abs(j[0]) > 1e-3 or abs(j[1]) > 1e-3 or abs(j[2]) > 1e-3
                    
                    # Helper to calculate leg angle for straight leg detection
                    def calc_leg_angle(hip, knee, ankle):
                        v1 = hip - knee
                        v2 = ankle - knee
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    CMU_AVG_HEIGHT = 24.0
                    
                    # Cascading scale detection for partial poses
                    full_height = None
                    char_scale = None
                    scale_method = "default"
                    
                    # 1. Try full height (head to ankles) - best
                    if is_valid(0) and (is_valid(10) or is_valid(13)):
                        head_y = char_joints[0][1]
                        
                        # Use straight leg detection if both legs visible
                        if is_valid(8) and is_valid(9) and is_valid(10) and is_valid(11) and is_valid(12) and is_valid(13):
                            l_hip, l_knee, l_ankle = char_joints[11], char_joints[12], char_joints[13]
                            r_hip, r_knee, r_ankle = char_joints[8], char_joints[9], char_joints[10]
                            
                            l_angle = calc_leg_angle(l_hip, l_knee, l_ankle)
                            r_angle = calc_leg_angle(r_hip, r_knee, r_ankle)
                            
                            straight_legs = []
                            if l_angle > 160:
                                straight_legs.append(l_ankle[1])
                            if r_angle > 160:
                                straight_legs.append(r_ankle[1])
                            
                            if straight_legs:
                                foot_y = sum(straight_legs) / len(straight_legs)
                            else:
                                foot_y = max(l_ankle[1], r_ankle[1])
                        else:
                            # Partial leg data - use what we have
                            l_ankle_y = char_joints[13][1] if is_valid(13) else char_joints[10][1]
                            r_ankle_y = char_joints[10][1] if is_valid(10) else char_joints[13][1]
                            foot_y = max(l_ankle_y, r_ankle_y)
                        
                        full_height = abs(foot_y - head_y)
                        if full_height > 10:
                            char_scale = full_height / CMU_AVG_HEIGHT
                            scale_method = "full_height"
                        else:
                            full_height = None
                    
                    # 2. Try torso (neck to hips) - good
                    if char_scale is None and is_valid(1) and (is_valid(8) or is_valid(11)):
                        neck = char_joints[1]
                        r_hip = char_joints[8] if is_valid(8) else char_joints[11]
                        l_hip = char_joints[11] if is_valid(11) else char_joints[8]
                        mid_hip = (r_hip + l_hip) / 2
                        torso_len = np.linalg.norm(neck - mid_hip)
                        
                        if torso_len > 3:
                            # Torso is ~40% of full height, so full = torso / 0.4 = torso * 2.5
                            full_height = torso_len * 2.5
                            char_scale = full_height / CMU_AVG_HEIGHT
                            scale_method = "torso"
                    
                    # 3. Try shoulder width - decent
                    if char_scale is None and is_valid(2) and is_valid(5):
                        r_shoulder = char_joints[2]
                        l_shoulder = char_joints[5]
                        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
                        
                        if shoulder_width > 2:
                            # Shoulder width is ~25% of height, so full = width * 4
                            full_height = shoulder_width * 4
                            char_scale = full_height / CMU_AVG_HEIGHT
                            scale_method = "shoulders"
                    
                    # 4. Fallback to defaults
                    if char_scale is None:
                        full_height = 200
                        char_scale = 1.0
                        scale_method = "default"
                    
                    print(f"[CMU] Scale method: {scale_method}, scale: {char_scale:.3f}")
                    
                    # Get position/depth from best available joint
                    if is_valid(1):
                        neck = char_joints[1]
                        center_x = neck[0]
                        depth_z = neck[2]
                    elif is_valid(2) and is_valid(5):
                        center_x = (char_joints[2][0] + char_joints[5][0]) / 2
                        depth_z = (char_joints[2][2] + char_joints[5][2]) / 2
                    else:
                        center_x = 256
                        depth_z = 800
                    
                    # Floor Y from lowest valid joint
                    valid_y = [char_joints[j][1] for j in range(18) if is_valid(j)]
                    floor_y = max(valid_y) if valid_y else 400
                    
                    ref_data.append({
                        'height': full_height,
                        'floor_y': floor_y,
                        'center_x': center_x,
                        'depth_z': depth_z,
                        'scale': char_scale
                    })
                else:
                    ref_data.append({'height': 200, 'floor_y': 400, 'center_x': 256, 'depth_z': 800, 'scale': 1.0})
        
        elif pose_sequence_in is not None and 'ref_data' in pose_sequence_in:
            # INHERIT FROM CHAIN
            ref_data = pose_sequence_in['ref_data']
            char_count = pose_sequence_in.get('char_count', 1)
        
        else:
            # DEFAULTS
            char_count = 1
            ref_data.append({'height': 200, 'floor_y': 400, 'center_x': 256, 'depth_z': 800, 'scale': 1.0})
        
        ref = ref_data[0]
        scale = ref['scale']
        
        # Initialize position tracking
        if last_position_x is not None:
            current_x = last_position_x
        else:
            current_x = ref['center_x']
        
        # Get first frame center for offset calculation
        first_frame = motion_data[0]
        first_center_x = (first_frame[2, 0] + first_frame[5, 0] + first_frame[8, 0] + first_frame[11, 0]) / 4
        
        # Track motion playback
        src_frame = 0
        direction = 1  # 1 = forward, -1 = reverse (for ping_pong)
        prev_center_x = first_center_x
        
        # Generate frames
        output_poses = list(input_poses)  # Start with input poses
        frames_to_generate = frame_count
        
        for frame_idx in range(frames_to_generate):
            # Get source frame with interpolation
            src_float = src_frame
            src_low = int(src_float)
            src_high = min(src_low + 1, motion_frames - 1)
            t = src_float - src_low
            
            # Handle loop modes
            if src_low >= motion_frames - 1:
                if loop_mode == "loop":
                    src_frame = 0
                    src_low = 0
                    src_high = 1
                    t = 0
                    # Reset position offset for seamless loop
                    prev_center_x = first_center_x
                elif loop_mode == "ping_pong":
                    direction *= -1
                    src_frame = motion_frames - 2
                    src_low = motion_frames - 2
                    src_high = motion_frames - 1
                    t = 0
                else:
                    # Hold last frame
                    src_low = motion_frames - 1
                    src_high = motion_frames - 1
                    t = 0
            elif src_low < 0:
                if loop_mode == "ping_pong":
                    direction *= -1
                    src_frame = 1
                    src_low = 0
                    src_high = 1
                    t = 0
                else:
                    src_low = 0
                    src_high = 0
                    t = 0
            
            # Interpolate pose
            pose = (1 - t) * motion_data[src_low] + t * motion_data[src_high]
            
            # Calculate current frame center
            frame_center_x = (pose[2, 0] + pose[5, 0] + pose[8, 0] + pose[11, 0]) / 4
            
            # Accumulate drift (negative because we mirror X)
            delta_x = (frame_center_x - prev_center_x) * scale * direction
            # current_x += delta_x
            prev_center_x = frame_center_x
            
            # Build output for all characters
            all_joints = []
            
            for char_idx in range(char_count):
                char_pose = pose.copy()
                char_ref = ref_data[char_idx] if char_idx < len(ref_data) else ref_data[0]
                    
# --- FACE SYNTHESIS START ---
                # CMU data maps index 0 to "Top of Head". OpenPose expects "Nose".
                
                raw_head_top = char_pose[0]
                neck = char_pose[1]
                
                # 1. Calculate Torso for Stable Scaling
                # The raw head-to-neck distance fluctuates in CMU data, causing "shrinking".
                # We use the torso length (Neck to Hips) as a stable reference for head size.
                r_hip = char_pose[8]
                l_hip = char_pose[11]
                mid_hip = (r_hip + l_hip) / 2
                spine_vec = neck - mid_hip
                torso_len = np.linalg.norm(spine_vec)
                
                # Fix head size to roughly 40% of torso length
                # This guarantees the head never shrinks, even if raw data collapses.
                stable_head_len = torso_len * 0.4
                
                # 2. Determine Orientation
                head_vec_raw = raw_head_top - neck
                raw_len = np.linalg.norm(head_vec_raw)
                
                # If raw head data is collapsed/garbage, align head with spine direction
                if raw_len < 0.01 or raw_len < (torso_len * 0.1):
                    up_dir = spine_vec / (torso_len + 1e-6)
                else:
                    up_dir = head_vec_raw / (raw_len + 1e-6)
                
                r_sh = char_pose[2]
                l_sh = char_pose[5]
                sh_vec = l_sh - r_sh
                right_dir = sh_vec / (np.linalg.norm(sh_vec) + 1e-6)
                
                # Forward vector
                fwd_dir = np.cross(right_dir, up_dir)
                fwd_dir = fwd_dir / (np.linalg.norm(fwd_dir) + 1e-6)
                
                # 3. Reposition Nose (Index 0)
                # Use the STABLE length for placement
                new_nose = neck + (up_dir * stable_head_len * 0.7) + (fwd_dir * stable_head_len * 0.25)
                char_pose[0] = new_nose 
                nose = new_nose
                
                # 4. Synthesize Eyes and Ears using STABLE length
                # Right Eye (14)
                char_pose[14] = nose + (right_dir * -0.15 * stable_head_len) + (up_dir * 0.15 * stable_head_len) - (fwd_dir * 0.05 * stable_head_len)
                # Left Eye (15)
                char_pose[15] = nose + (right_dir * 0.15 * stable_head_len) + (up_dir * 0.15 * stable_head_len) - (fwd_dir * 0.05 * stable_head_len)
                # Right Ear (16)
                char_pose[16] = nose + (right_dir * -0.4 * stable_head_len) + (up_dir * 0.0 * stable_head_len) - (fwd_dir * 0.3 * stable_head_len)
                # Left Ear (17)
                char_pose[17] = nose + (right_dir * 0.4 * stable_head_len) + (up_dir * 0.0 * stable_head_len) - (fwd_dir * 0.3 * stable_head_len)
                # --- FACE SYNTHESIS END ---

                # Get body center
                shoulder_center = (char_pose[2] + char_pose[5]) / 2
                hip_center = (char_pose[8] + char_pose[11]) / 2
                pose_center_x = (shoulder_center[0] + hip_center[0]) / 2
                pose_center_y = (shoulder_center[1] + hip_center[1]) / 2
                
                # Scale around center (but don't re-center X - use accumulated position)
# WIDTH CORRECTION:
                # CMU data is often too "wide" for standard/anime proportions.
                # We multiply the X offset by 0.75 - 0.85 to slim the shoulders/hips.
                width_factor = 0.8
                
                char_pose[:, 0] = (char_pose[:, 0] - pose_center_x) * scale * width_factor + current_x
                char_pose[:, 1] = (char_pose[:, 1] - pose_center_y) * scale
                char_pose[:, 2] = char_ref['depth_z']
                
                # Flip Y (CMU Y-up to screen Y-down)
                char_pose[:, 1] = -char_pose[:, 1]
                
                # Floor anchor
                pose_floor_y = np.max(char_pose[:, 1])
                floor_offset = char_ref['floor_y'] - pose_floor_y
                char_pose[:, 1] = char_pose[:, 1] + floor_offset
                
                all_joints.append(char_pose.astype(np.float32))
            
            frame_joints = np.concatenate(all_joints, axis=0)
            output_poses.append(frame_joints)
            
            # Advance source frame
            src_frame += direction * time_scale
        
        # Blend transition from previous sequence
        if last_pose is not None and transition_frames > 0:
            blend_start = start_frame
            blend_end = min(start_frame + transition_frames, len(output_poses))
            
            for i in range(blend_start, blend_end):
                t = (i - blend_start) / transition_frames
                t = t * t * (3 - 2 * t)  # Smooth easing
                output_poses[i] = (1 - t) * last_pose + t * output_poses[i]
        
        # Blend from reference pose if this is first in chain
        elif scail_pose is not None and transition_frames > 0 and start_frame == 0:
            ref_joints_np = scail_pose['joints'].numpy()
            blend_frames = min(transition_frames, len(output_poses))
            
            for i in range(blend_frames):
                t = i / blend_frames
                t = t * t * (3 - 2 * t)
                output_poses[i] = (1 - t) * ref_joints_np + t * output_poses[i]
        
        # OpenPose 18 limb sequence (matching SCAIL nodes.py exactly)
        limb_seq = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
        ]
        
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        # RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        
        # Build full limb_seq and colors for all characters
        full_limb_seq = []
        full_colors = []
        for i in range(char_count):
            offset = i * 18
            for limb in limb_seq:
                full_limb_seq.append((limb[0] + offset, limb[1] + offset))
            full_colors.extend(colors)
        
        # Build output sequence
        pose_sequence = {
            'poses': output_poses,
            'limb_seq': full_limb_seq,
            'bone_colors': full_colors,
            'frame_count': len(output_poses),
            'last_position_x': current_x,
            'char_count': char_count,
            'ref_data': ref_data
        }
        
        print(f"[CMU] Output keys: {pose_sequence.keys()}")
        print(f"[CMU] Poses count: {len(output_poses)}")
        
        return (pose_sequence, len(output_poses), motion_desc)


class SCAILCMUSequenceToSCAILPose:
    """Convert CMU pose sequence to frame-by-frame SCAIL_POSE for rendering."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_sequence": ("SCAIL_POSE_SEQUENCE",),
                "image_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "image_height": ("INT", {"default": 768, "min": 64, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("pose_sequence",)
    FUNCTION = "convert"
    CATEGORY = "SCAIL/Motion"
    
    def convert(self, pose_sequence, image_width, image_height):
        poses = pose_sequence['poses']
        char_count = pose_sequence.get('char_count', 1)
        
        # Stack all poses into tensor
        pose_array = np.stack(poses, axis=0)
        joints_tensor = torch.from_numpy(pose_array).float()
        
        # OpenPose 18 limb sequence (matching SCAIL nodes.py exactly)
        limb_seq = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
        ]
        
        # RGBA colors matching SCAIL nodes.py (converted to RGB tuples 0-255)
        colors = [
            [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
            [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
            [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
            [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
            [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
            [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
        ]
        
        # Build full limb_seq and colors for all characters
        full_limb_seq = []
        full_colors = []
        for i in range(char_count):
            offset = i * 18
            for limb in limb_seq:
                full_limb_seq.append((limb[0] + offset, limb[1] + offset))
            full_colors.extend(colors)
        
        return ({
            'joints': joints_tensor,
            'limb_seq': full_limb_seq,
            'colors': full_colors,
            'char_count': char_count,
            'frame_count': len(poses),
            'image_width': image_width,
            'image_height': image_height,
        },)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SCAILCMULibraryLoader": SCAILCMULibraryLoader,
    "SCAILCMUMotion": SCAILCMUMotion,
    "SCAILCMUSequenceToSCAILPose": SCAILCMUSequenceToSCAILPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILCMULibraryLoader": "SCAIL CMU Library Loader",
    "SCAILCMUMotion": "SCAIL CMU Motion",
    "SCAILCMUSequenceToSCAILPose": "SCAIL CMU Sequence to Pose",
}
