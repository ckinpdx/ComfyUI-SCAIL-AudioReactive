"""
FBX Motion Nodes for ComfyUI

Provides motion capture playback from FBX files using Blender headless extraction.
Outputs SCAIL_POSE_SEQUENCE compatible with SCAIL Pose Renderer.
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
import folder_paths  # ComfyUI's folder management

# Register FBX input folder with ComfyUI
FBX_FOLDER_NAME = "fbx_animations"
fbx_folder = os.path.join(folder_paths.get_input_directory(), FBX_FOLDER_NAME)
os.makedirs(fbx_folder, exist_ok=True)

# OpenPose 18 limb sequence (matching SCAIL nodes.py exactly)
LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# RGBA colors matching SCAIL nodes.py BONE_COLORS exactly
BONE_COLORS = [
    [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
    [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
    [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
    [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
    [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
    [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
]


def find_blender_installations():
    """
    Find Blender installations in standard locations.
    Returns list of (version, path) tuples sorted by version descending.
    """
    installations = []
    
    # Windows paths
    if sys.platform == "win32":
        base_paths = [
            Path("C:/Program Files/Blender Foundation"),
            Path("C:/Program Files (x86)/Blender Foundation"),
            Path(os.path.expanduser("~/AppData/Local/Blender Foundation")),
        ]
        
        for base in base_paths:
            if base.exists():
                for folder in base.iterdir():
                    if folder.is_dir() and folder.name.startswith("Blender"):
                        # Extract version from folder name like "Blender 4.5" or "Blender"
                        exe_path = folder / "blender.exe"
                        if exe_path.exists():
                            # Try to extract version number
                            parts = folder.name.split()
                            version = parts[1] if len(parts) > 1 else "0.0"
                            installations.append((version, str(exe_path)))
    
    # Linux paths
    elif sys.platform == "linux":
        possible_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            os.path.expanduser("~/blender/blender"),
        ]
        
        # Also check /opt for versioned installations
        opt_blender = Path("/opt")
        if opt_blender.exists():
            for folder in opt_blender.iterdir():
                if folder.is_dir() and "blender" in folder.name.lower():
                    exe = folder / "blender"
                    if exe.exists():
                        possible_paths.append(str(exe))
        
        for path in possible_paths:
            if os.path.exists(path):
                installations.append(("auto", path))
    
    # macOS paths
    elif sys.platform == "darwin":
        app_path = Path("/Applications/Blender.app/Contents/MacOS/Blender")
        if app_path.exists():
            installations.append(("auto", str(app_path)))
        
        # Check for versioned apps
        apps = Path("/Applications")
        for app in apps.glob("Blender*.app"):
            exe = app / "Contents/MacOS/Blender"
            if exe.exists():
                version = app.name.replace("Blender", "").replace(".app", "").strip()
                installations.append((version or "auto", str(exe)))
    
    # Sort by version descending (newest first)
    installations.sort(key=lambda x: x[0], reverse=True)
    
    return installations


def get_blender_choices():
    """Get list of Blender installations for dropdown."""
    installations = find_blender_installations()
    
    if not installations:
        return ["NOT FOUND - Install Blender"]
    
    choices = []
    for version, path in installations:
        # Create a readable label
        if version == "auto":
            label = f"Blender ({path})"
        else:
            label = f"Blender {version}"
        choices.append(label)
    
    return choices


def get_blender_path_from_choice(choice):
    """Convert dropdown choice back to actual path."""
    installations = find_blender_installations()
    
    for version, path in installations:
        if version == "auto":
            label = f"Blender ({path})"
        else:
            label = f"Blender {version}"
        
        if label == choice:
            return path
    
    return None


class SCAILMixamoFBX:
    """
    Load FBX animation and convert to SCAIL pose sequence.
    Uses Blender headless to extract joint positions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        blender_choices = get_blender_choices()
        
        # Scan FBX folder for available files
        fbx_files = []
        if os.path.exists(fbx_folder):
            for f in os.listdir(fbx_folder):
                if f.lower().endswith('.fbx'):
                    fbx_files.append(f)
        
        if not fbx_files:
            fbx_files = ["NO FBX FILES - Add to ComfyUI/input/fbx_animations/"]
        
        return {
            "required": {
                "fbx_file": (sorted(fbx_files), {"tooltip": "FBX file from ComfyUI/input/fbx_animations/ folder"}),
                "blender_version": (blender_choices, {"default": blender_choices[0] if blender_choices else "NOT FOUND"}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Output FPS (will resample if different from source)"}),
                "transition_frames": ("INT", {"default": 12, "min": 0, "max": 60, "tooltip": "Blend frames from reference pose"}),
            },
            "optional": {
                "scail_pose": ("SCAIL_POSE", {"tooltip": "Reference pose for scaling/positioning"}),
                "pose_sequence_in": ("SCAIL_POSE_SEQUENCE", {"tooltip": "Chain from previous motion"}),
            }
        }
    
    # Tell ComfyUI to refresh dropdown when files change
    @classmethod
    def IS_CHANGED(cls, fbx_file, **kwargs):
        fbx_path = os.path.join(fbx_folder, fbx_file)
        if os.path.exists(fbx_path):
            return os.path.getmtime(fbx_path)
        return float("nan")
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("pose_sequence", "frame_count_out", "motion_info")
    FUNCTION = "load_fbx"
    CATEGORY = "SCAIL/Motion"
    
    def load_fbx(self, fbx_file, blender_version, target_fps, transition_frames,
                 scail_pose=None, pose_sequence_in=None):
        
        # Build full path from selected filename
        fbx_path = os.path.join(fbx_folder, fbx_file)
        
        # Validate Blender path
        blender_path = get_blender_path_from_choice(blender_version)
        if not blender_path or not os.path.exists(blender_path):
            raise RuntimeError(f"Blender not found. Please install Blender and restart ComfyUI.")
        
        # Validate FBX file
        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}. Add FBX files to ComfyUI/input/fbx_animations/")
        
        # Get the extraction script path (same directory as this file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        extract_script = os.path.join(script_dir, "blender_extract.py")
        
        if not os.path.exists(extract_script):
            raise RuntimeError(f"Blender extraction script not found: {extract_script}")
        
        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_json = tmp.name
        
        try:
            # Run Blender headless
            print(f"[FBX] Running Blender extraction: {fbx_path}")
            
            env = os.environ.copy()
            env["FBX_PATH"] = fbx_path
            env["OUTPUT_JSON"] = output_json
            
            cmd = [blender_path, "--background", "--python", extract_script]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                print(f"[FBX] Blender stderr: {result.stderr}")
                raise RuntimeError(f"Blender extraction failed: {result.stderr[:500]}")
            
            # Parse output JSON
            if not os.path.exists(output_json):
                raise RuntimeError("Blender extraction produced no output")
            
            with open(output_json, 'r') as f:
                motion_data = json.load(f)
            
        finally:
            # Cleanup temp file
            if os.path.exists(output_json):
                os.unlink(output_json)
        
        # Convert to numpy
        source_fps = motion_data["fps"]
        frames = np.array(motion_data["frames"], dtype=np.float32)  # [num_frames, 18, 3]
        
        print(f"[FBX] Loaded {len(frames)} frames @ {source_fps} fps")
        
        # Validate extracted data - check if joints are all zeros (incompatible FBX)
        first_frame = frames[0]
        non_zero_joints = np.sum(np.abs(first_frame) > 1e-6)
        if non_zero_joints < 10:  # Less than 10 non-zero values means extraction failed
            print(f"[FBX] ERROR: Incompatible FBX file - bone extraction failed!")
            print(f"[FBX] ERROR: Only {non_zero_joints}/54 joint values extracted.")
            print(f"[FBX] ERROR: This FBX may use non-standard bone naming.")
            raise RuntimeError(f"Incompatible FBX file: {fbx_file}. Bone names not recognized. Try a different FBX export.")
        
        # Time scaling
        time_scale = source_fps / target_fps
        output_frame_count = int(len(frames) / time_scale)
        
        # Get reference data for scaling/positioning
        if pose_sequence_in is not None:
            # Chaining from previous motion
            input_poses = pose_sequence_in['poses']
            ref_data = pose_sequence_in.get('ref_data', [{'height': 200, 'floor_y': 400, 'center_x': 0, 'depth_z': 800, 'scale': 1.0}])
            char_count = pose_sequence_in.get('char_count', 1)
            last_pose = input_poses[-1] if len(input_poses) > 0 else None
        elif scail_pose is not None:
            input_poses = []
            ref_joints = scail_pose['joints'].numpy()
            char_count = scail_pose.get('char_count', 1)
            last_pose = None
            
            # Build reference data per character
            ref_data = []
            for i in range(char_count):
                start_idx = i * 18
                end_idx = start_idx + 18
                if end_idx <= len(ref_joints):
                    char_joints = ref_joints[start_idx:end_idx]
                    
                    # Helper to check if joint is valid (not zero/origin)
                    def is_valid(joint_idx):
                        j = char_joints[joint_idx]
                        return abs(j[0]) > 1e-3 or abs(j[1]) > 1e-3 or abs(j[2]) > 1e-3
                    
                    # Source motion reference
                    src_first = frames[0]
                    
                    # Cascading scale detection for partial poses
                    full_height = None
                    char_scale = None
                    scale_method = "default"
                    
                    # 1. Try full height (head to ankles) - best
                    if is_valid(0) and (is_valid(10) or is_valid(13)):
                        head_y = char_joints[0][1]
                        l_ankle_y = char_joints[13][1] if is_valid(13) else char_joints[10][1]
                        r_ankle_y = char_joints[10][1] if is_valid(10) else char_joints[13][1]
                        foot_y = max(l_ankle_y, r_ankle_y)
                        full_height = abs(foot_y - head_y)
                        
                        src_head_y = src_first[0, 1]
                        src_foot_y = max(src_first[10, 1], src_first[13, 1])
                        src_height = abs(src_foot_y - src_head_y)
                        
                        if full_height > 10 and src_height > 0:
                            char_scale = full_height / src_height
                            scale_method = "full_height"
                        else:
                            full_height = None  # Reset so next method tries
                    
                    # 2. Try torso (neck to hips) - good
                    if char_scale is None and is_valid(1) and (is_valid(8) or is_valid(11)):
                        neck = char_joints[1]
                        r_hip = char_joints[8] if is_valid(8) else char_joints[11]
                        l_hip = char_joints[11] if is_valid(11) else char_joints[8]
                        mid_hip = (r_hip + l_hip) / 2
                        torso_len = np.linalg.norm(neck - mid_hip)
                        
                        src_neck = src_first[1]
                        src_mid_hip = (src_first[8] + src_first[11]) / 2
                        src_torso = np.linalg.norm(src_neck - src_mid_hip)
                        
                        if torso_len > 5 and src_torso > 0:
                            char_scale = torso_len / src_torso
                            full_height = torso_len * 2.5  # Estimate full height
                            scale_method = "torso"
                    
                    # 3. Try shoulder width - decent
                    if char_scale is None and is_valid(2) and is_valid(5):
                        r_shoulder = char_joints[2]
                        l_shoulder = char_joints[5]
                        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
                        
                        src_shoulder_width = np.linalg.norm(src_first[5] - src_first[2])
                        
                        if shoulder_width > 5 and src_shoulder_width > 0:
                            char_scale = shoulder_width / src_shoulder_width
                            full_height = shoulder_width * 4  # Estimate full height
                            scale_method = "shoulders"
                    
                    # 4. Fallback to defaults
                    if char_scale is None:
                        full_height = 200
                        char_scale = 1.0
                        scale_method = "default"
                    
                    print(f"[FBX] Scale method: {scale_method}, scale: {char_scale:.3f}")
                    
                    # Get position/depth from best available joint
                    if is_valid(1):
                        neck = char_joints[1]
                        center_x = neck[0]
                        depth_z = neck[2]
                    elif is_valid(2) and is_valid(5):
                        center_x = (char_joints[2][0] + char_joints[5][0]) / 2
                        depth_z = (char_joints[2][2] + char_joints[5][2]) / 2
                    else:
                        center_x = 0
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
                    ref_data.append({'height': 200, 'floor_y': 400, 'center_x': 0, 'depth_z': 800, 'scale': 1.0})
        else:
            # No reference - use defaults
            input_poses = []
            char_count = 1
            ref_data = [{'height': 200, 'floor_y': 400, 'center_x': 0, 'depth_z': 800, 'scale': 1.0}]
            last_pose = None
        
        ref = ref_data[0]
        scale = ref['scale']
        
        # Generate output frames
        output_poses = list(input_poses)
        
        for frame_idx in range(output_frame_count):
            # Map to source frame with interpolation
            src_float = frame_idx * time_scale
            src_low = int(src_float)
            src_high = min(src_low + 1, len(frames) - 1)
            t = src_float - src_low
            
            # Interpolate
            pose = (1 - t) * frames[src_low] + t * frames[src_high]
            
            # Build output for all characters
            all_joints = []
            
            for char_idx in range(char_count):
                char_pose = pose.copy()
                char_ref = ref_data[char_idx] if char_idx < len(ref_data) else ref_data[0]
                char_scale = char_ref['scale']
                
                # Get body center
                shoulder_center = (char_pose[2] + char_pose[5]) / 2
                hip_center = (char_pose[8] + char_pose[11]) / 2
                pose_center_x = (shoulder_center[0] + hip_center[0]) / 2
                pose_center_y = (shoulder_center[1] + hip_center[1]) / 2
                pose_center_z = (shoulder_center[2] + hip_center[2]) / 2
                
                # Scale and position - preserve relative Z movement for perspective
                char_pose[:, 0] = (char_pose[:, 0] - pose_center_x) * char_scale + char_ref['center_x']
                char_pose[:, 1] = (char_pose[:, 1] - pose_center_y) * char_scale
                # Preserve relative Z without scaling - just offset from reference depth
                char_pose[:, 2] = (char_pose[:, 2] - pose_center_z) + char_ref['depth_z']
                
                # Flip Y (Blender Z-up to screen Y-down)
                # Blender uses Z-up, Y-forward. We need to convert.
                # Screen Y increases downward
                char_pose[:, 1] = -char_pose[:, 1]
                
                # Floor anchor
                pose_floor_y = np.max(char_pose[:, 1])
                floor_offset = char_ref['floor_y'] - pose_floor_y
                char_pose[:, 1] = char_pose[:, 1] + floor_offset
                
                all_joints.append(char_pose.astype(np.float32))
            
            frame_joints = np.concatenate(all_joints, axis=0)
            output_poses.append(frame_joints)
        
        # Blend transition from previous sequence
        if last_pose is not None and transition_frames > 0:
            start_frame = len(input_poses)
            blend_end = min(start_frame + transition_frames, len(output_poses))
            
            for i in range(start_frame, blend_end):
                t = (i - start_frame) / transition_frames
                t = t * t * (3 - 2 * t)  # Smooth easing
                output_poses[i] = (1 - t) * last_pose + t * output_poses[i]
        
        # Blend from reference pose if first in chain
        elif scail_pose is not None and transition_frames > 0:
            ref_joints_np = scail_pose['joints'].numpy()
            blend_frames = min(transition_frames, len(output_poses))
            
            for i in range(blend_frames):
                t = i / blend_frames
                t = t * t * (3 - 2 * t)
                output_poses[i] = (1 - t) * ref_joints_np + t * output_poses[i]
        
        # Build limb_seq and colors for all characters
        full_limb_seq = []
        full_colors = []
        for i in range(char_count):
            offset = i * 18
            for limb in LIMB_SEQ:
                full_limb_seq.append((limb[0] + offset, limb[1] + offset))
            full_colors.extend(BONE_COLORS)
        
        # Build output
        pose_sequence = {
            'poses': output_poses,
            'limb_seq': full_limb_seq,
            'bone_colors': full_colors,
            'frame_count': len(output_poses),
            'char_count': char_count,
            'ref_data': ref_data
        }
        
        motion_info = f"FBX: {fbx_file} ({motion_data['frame_count']} frames @ {source_fps}fps -> {len(output_poses)} @ {target_fps}fps)"
        print(f"[FBX] {motion_info}")
        
        return (pose_sequence, len(output_poses), motion_info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SCAILMixamoFBX": SCAILMixamoFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILMixamoFBX": "SCAIL FBX Motion",
}
