"""
Blender Headless FBX Extraction Script

This script runs INSIDE Blender via subprocess.
It extracts joint positions from FBX files and outputs JSON.

Usage (called by mixamo_nodes.py):
    blender --background --python blender_extract_mixamo.py

Environment variables:
    FBX_PATH: Path to input FBX file
    OUTPUT_JSON: Path to output JSON file
"""

import bpy
import json
import os
import sys
import math
from mathutils import Vector, Matrix

# FBX bone names to OpenPose 18 joint indices
# Supports "mixamorig:" prefixed bones (common in rigged FBX exports)
# OpenPose: 0=nose, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist,
#           5=l_shoulder, 6=l_elbow, 7=l_wrist, 8=r_hip, 9=r_knee,
#           10=r_ankle, 11=l_hip, 12=l_knee, 13=l_ankle,
#           14=r_eye, 15=l_eye, 16=r_ear, 17=l_ear

MIXAMO_TO_OPENPOSE = {
    # Head - will be used to synthesize face
    "mixamorig:Head": "head",  # Special - used for face synthesis
    "mixamorig:Neck": 1,
    
    # Right arm
    "mixamorig:RightShoulder": 2,
    "mixamorig:RightArm": 2,  # Fallback
    "mixamorig:RightForeArm": 3,
    "mixamorig:RightHand": 4,
    
    # Left arm  
    "mixamorig:LeftShoulder": 5,
    "mixamorig:LeftArm": 5,  # Fallback
    "mixamorig:LeftForeArm": 6,
    "mixamorig:LeftHand": 7,
    
    # Right leg
    "mixamorig:RightUpLeg": 8,
    "mixamorig:RightLeg": 9,
    "mixamorig:RightFoot": 10,
    
    # Left leg
    "mixamorig:LeftUpLeg": 11,
    "mixamorig:LeftLeg": 12,
    "mixamorig:LeftFoot": 13,
    
    # Hips for reference
    "mixamorig:Hips": "hips",
}

# Alternative naming (some Mixamo exports don't have "mixamorig:" prefix)
MIXAMO_ALT_NAMES = {
    "Head": "mixamorig:Head",
    "Neck": "mixamorig:Neck",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand": "mixamorig:RightHand",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "LeftHand": "mixamorig:LeftHand",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "Hips": "mixamorig:Hips",
}


def get_bone_world_position(armature, bone_name, frame):
    """Get world position of a bone at a specific frame."""
    bpy.context.scene.frame_set(frame)
    
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        # Try alternative name
        alt_name = MIXAMO_ALT_NAMES.get(bone_name.replace("mixamorig:", ""))
        if alt_name:
            pose_bone = armature.pose.bones.get(alt_name.replace("mixamorig:", ""))
        if pose_bone is None:
            return None
    
    # Get world matrix and extract position
    world_matrix = armature.matrix_world @ pose_bone.matrix
    return world_matrix.translation.copy()


def find_armature():
    """Find the armature object in the scene."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def get_available_bones(armature):
    """Get list of available bone names."""
    return [bone.name for bone in armature.pose.bones]


def normalize_bone_name(name):
    """Try to match bone name to Mixamo standard."""
    if name in MIXAMO_TO_OPENPOSE:
        return name
    
    # Try with prefix
    prefixed = f"mixamorig:{name}"
    if prefixed in MIXAMO_TO_OPENPOSE:
        return prefixed
    
    # Try without prefix (mixamorig:)
    if name.startswith("mixamorig:"):
        unprefixed = name[10:]
        normalized = f"mixamorig:{unprefixed}"
        if normalized in MIXAMO_TO_OPENPOSE:
            return normalized
    
    # Try without prefix (mixamorig1: - alternate naming with "1")
    if name.startswith("mixamorig1:"):
        unprefixed = name[11:]
        normalized = f"mixamorig:{unprefixed}"
        if normalized in MIXAMO_TO_OPENPOSE:
            return normalized
    
    # Try without prefix (mixamorig2:, mixamorig3:, etc - just in case)
    import re
    match = re.match(r'mixamorig\d+:(.*)', name)
    if match:
        unprefixed = match.group(1)
        normalized = f"mixamorig:{unprefixed}"
        if normalized in MIXAMO_TO_OPENPOSE:
            return normalized
    
    return None


def extract_frame(armature, frame, bone_map):
    """Extract OpenPose 18-joint positions for a single frame."""
    # Initialize 18 joints with zeros
    joints = [[0.0, 0.0, 0.0] for _ in range(18)]
    
    # Storage for special bones needed for face synthesis
    head_pos = None
    neck_pos = None
    r_shoulder_pos = None
    l_shoulder_pos = None
    r_hip_pos = None
    l_hip_pos = None
    
    # Extract mapped bones
    for mixamo_name, openpose_idx in bone_map.items():
        pos = get_bone_world_position(armature, mixamo_name, frame)
        if pos is None:
            continue
        
        # Handle special markers
        if openpose_idx == "head":
            head_pos = pos
        elif openpose_idx == "hips":
            # Use hips to help calculate mid-hip if needed
            pass
        elif isinstance(openpose_idx, int):
            # Blender Z-up → Screen Y-vertical: swap Y and Z
            joints[openpose_idx] = [pos.x, pos.z, pos.y]
            
            # Store key positions for face synthesis
            if openpose_idx == 1:
                neck_pos = pos
            elif openpose_idx == 2:
                r_shoulder_pos = pos
            elif openpose_idx == 5:
                l_shoulder_pos = pos
            elif openpose_idx == 8:
                r_hip_pos = pos
            elif openpose_idx == 11:
                l_hip_pos = pos
    
    # Synthesize face keypoints (nose, eyes, ears) from head/neck
    if head_pos and neck_pos and r_shoulder_pos and l_shoulder_pos:
        # Calculate torso length for stable head sizing
        if r_hip_pos and l_hip_pos:
            mid_hip = (r_hip_pos + l_hip_pos) / 2
            spine_vec = neck_pos - mid_hip
            torso_len = spine_vec.length
        else:
            torso_len = (neck_pos - head_pos).length * 2.5
        
        stable_head_len = torso_len * 0.4
        
        # Orientation vectors
        head_vec = head_pos - neck_pos
        raw_len = head_vec.length
        
        if raw_len < 0.01 or raw_len < (torso_len * 0.1):
            up_dir = spine_vec.normalized() if r_hip_pos and l_hip_pos else Vector((0, 0, 1))
        else:
            up_dir = head_vec.normalized()
        
        sh_vec = l_shoulder_pos - r_shoulder_pos
        right_dir = sh_vec.normalized()
        
        fwd_dir = right_dir.cross(up_dir).normalized()
        
        # Synthesize nose (index 0)
        nose = neck_pos + (up_dir * stable_head_len * 0.7) + (fwd_dir * stable_head_len * 0.25)
        joints[0] = [nose.x, nose.z, nose.y]
        
        # Right eye (index 14)
        r_eye = nose + (right_dir * -0.15 * stable_head_len) + (up_dir * 0.15 * stable_head_len) - (fwd_dir * 0.05 * stable_head_len)
        joints[14] = [r_eye.x, r_eye.z, r_eye.y]
        
        # Left eye (index 15)
        l_eye = nose + (right_dir * 0.15 * stable_head_len) + (up_dir * 0.15 * stable_head_len) - (fwd_dir * 0.05 * stable_head_len)
        joints[15] = [l_eye.x, l_eye.z, l_eye.y]
        
        # Right ear (index 16)
        r_ear = nose + (right_dir * -0.4 * stable_head_len) - (fwd_dir * 0.3 * stable_head_len)
        joints[16] = [r_ear.x, r_ear.z, r_ear.y]
        
        # Left ear (index 17)
        l_ear = nose + (right_dir * 0.4 * stable_head_len) - (fwd_dir * 0.3 * stable_head_len)
        joints[17] = [l_ear.x, l_ear.z, l_ear.y]
    
    return joints


def main():
    # Get paths from environment
    fbx_path = os.environ.get("FBX_PATH")
    output_path = os.environ.get("OUTPUT_JSON")
    
    if not fbx_path or not output_path:
        print("ERROR: FBX_PATH and OUTPUT_JSON environment variables required")
        sys.exit(1)
    
    if not os.path.exists(fbx_path):
        print(f"ERROR: FBX file not found: {fbx_path}")
        sys.exit(1)
    
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import FBX
    print(f"Importing FBX: {fbx_path}")
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except Exception as e:
        print(f"ERROR: Failed to import FBX: {e}")
        sys.exit(1)
    
    # Find armature
    armature = find_armature()
    if armature is None:
        print("ERROR: No armature found in FBX")
        sys.exit(1)
    
    print(f"Found armature: {armature.name}")
    
    # Get available bones and build mapping
    available_bones = get_available_bones(armature)
    print(f"Available bones: {available_bones[:10]}...")  # Print first 10
    
    # Build bone mapping
    bone_map = {}
    for bone_name in available_bones:
        normalized = normalize_bone_name(bone_name)
        if normalized and normalized in MIXAMO_TO_OPENPOSE:
            bone_map[bone_name] = MIXAMO_TO_OPENPOSE[normalized]
    
    print(f"Mapped {len(bone_map)} bones")
    
    # Get animation range from the actual action, not scene defaults
    scene = bpy.context.scene
    fps = scene.render.fps
    
    # Try to get frame range from armature's action
    frame_start = None
    frame_end = None
    
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start, frame_end = action.frame_range
        frame_start = int(frame_start)
        frame_end = int(frame_end)
        print(f"Using action '{action.name}' frame range: {frame_start}-{frame_end}")
    
    # Fallback: check all actions in the file
    if frame_start is None and bpy.data.actions:
        for action in bpy.data.actions:
            f_start, f_end = action.frame_range
            print(f"Found action '{action.name}': frames {int(f_start)}-{int(f_end)}")
            if frame_start is None or f_start < frame_start:
                frame_start = int(f_start)
            if frame_end is None or f_end > frame_end:
                frame_end = int(f_end)
        
        # Assign the first action to the armature if none assigned
        if armature.animation_data is None:
            armature.animation_data_create()
        if armature.animation_data.action is None and bpy.data.actions:
            armature.animation_data.action = bpy.data.actions[0]
            print(f"Assigned action '{bpy.data.actions[0].name}' to armature")
    
    # Final fallback to scene range
    if frame_start is None:
        frame_start = int(scene.frame_start)
        frame_end = int(scene.frame_end)
        print(f"Using scene frame range (fallback): {frame_start}-{frame_end}")
    
    print(f"Animation: frames {frame_start}-{frame_end} @ {fps} fps")
    
    # Extract all frames
    frames_data = []
    for frame in range(frame_start, frame_end + 1):
        joints = extract_frame(armature, frame, bone_map)
        frames_data.append(joints)
    
    # Build output
    output = {
        "fps": fps,
        "frame_count": len(frames_data),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "joints_per_frame": 18,
        "frames": frames_data,
        "joint_names": [
            "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
            "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
            "r_ankle", "l_hip", "l_knee", "l_ankle",
            "r_eye", "l_eye", "r_ear", "l_ear"
        ]
    }
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"Extracted {len(frames_data)} frames to {output_path}")


if __name__ == "__main__":
    main()
