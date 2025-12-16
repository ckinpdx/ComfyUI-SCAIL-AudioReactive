import torch
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

from .render_3d.taichi_cylinder import render_whole

# ============================================================================
# SKELETON DEFINITIONS
# ============================================================================

# COCO format joint indices (18 joints)
JOINT_NAMES = [
    "nose",        # 0
    "neck",        # 1
    "r_shoulder",  # 2
    "r_elbow",     # 3
    "r_wrist",     # 4
    "l_shoulder",  # 5
    "l_elbow",     # 6
    "l_wrist",     # 7
    "r_hip",       # 8
    "r_knee",      # 9
    "r_ankle",     # 10
    "l_hip",       # 11
    "l_knee",      # 12
    "l_ankle",     # 13
    "r_eye",       # 14
    "l_eye",       # 15
    "r_ear",       # 16
    "l_ear",       # 17
]

# Bone connections: (start_joint, end_joint)
LIMB_SEQ = [
    (1, 2),    # 0: Neck -> R. Shoulder
    (1, 5),    # 1: Neck -> L. Shoulder
    (2, 3),    # 2: R. Shoulder -> R. Elbow
    (3, 4),    # 3: R. Elbow -> R. Wrist
    (5, 6),    # 4: L. Shoulder -> L. Elbow
    (6, 7),    # 5: L. Elbow -> L. Wrist
    (1, 8),    # 6: Neck -> R. Hip
    (8, 9),    # 7: R. Hip -> R. Knee
    (9, 10),   # 8: R. Knee -> R. Ankle
    (1, 11),   # 9: Neck -> L. Hip
    (11, 12),  # 10: L. Hip -> L. Knee
    (12, 13),  # 11: L. Knee -> L. Ankle
    (1, 0),    # 12: Neck -> Nose
    (0, 14),   # 13: Nose -> R. Eye
    (14, 16),  # 14: R. Eye -> R. Ear
    (0, 15),   # 15: Nose -> L. Eye
    (15, 17),  # 16: L. Eye -> L. Ear
]

# Parent joint for each joint (for hierarchical transforms)
JOINT_PARENTS = {
    0: 1,    # nose -> neck
    1: -1,   # neck (root)
    2: 1,    # r_shoulder -> neck
    3: 2,    # r_elbow -> r_shoulder
    4: 3,    # r_wrist -> r_elbow
    5: 1,    # l_shoulder -> neck
    6: 5,    # l_elbow -> l_shoulder
    7: 6,    # l_wrist -> l_elbow
    8: 1,    # r_hip -> neck
    9: 8,    # r_knee -> r_hip
    10: 9,   # r_ankle -> r_knee
    11: 1,   # l_hip -> neck
    12: 11,  # l_knee -> l_hip
    13: 12,  # l_ankle -> l_knee
    14: 0,   # r_eye -> nose
    15: 0,   # l_eye -> nose
    16: 14,  # r_ear -> r_eye
    17: 15,  # l_ear -> l_eye
}

# Default colors for bones (RGBA, 0-1 range)
BONE_COLORS = [
    [1.0, 0.15, 0.15, 0.8],   # Neck -> R. Shoulder (Red)
    [0.15, 1.0, 1.0, 0.8],    # Neck -> L. Shoulder (Cyan)
    [1.0, 0.43, 0.15, 0.8],   # R. Shoulder -> R. Elbow (Orange)
    [1.0, 0.72, 0.15, 0.8],   # R. Elbow -> R. Wrist (Golden)
    [0.15, 0.72, 1.0, 0.8],   # L. Shoulder -> L. Elbow (Sky Blue)
    [0.15, 0.43, 1.0, 0.8],   # L. Elbow -> L. Wrist (Blue)
    [0.75, 1.0, 0.15, 0.8],   # Neck -> R. Hip (Yellow-Green)
    [0.15, 1.0, 0.15, 0.8],   # R. Hip -> R. Knee (Green)
    [0.15, 1.0, 0.43, 0.8],   # R. Knee -> R. Ankle (Light Green)
    [0.15, 0.15, 1.0, 0.8],   # Neck -> L. Hip (Blue)
    [0.43, 0.15, 1.0, 0.8],   # L. Hip -> L. Knee (Purple-Blue)
    [0.72, 0.15, 1.0, 0.8],   # L. Knee -> L. Ankle (Purple)
    [0.65, 0.65, 0.65, 0.8],  # Neck -> Nose (Grey)
    [1.0, 0.15, 0.72, 0.8],   # Nose -> R. Eye (Pink)
    [0.32, 0.15, 1.0, 0.8],   # R. Eye -> R. Ear (Violet)
    [1.0, 0.15, 0.72, 0.8],   # Nose -> L. Eye (Pink)
    [0.32, 0.15, 1.0, 0.8],   # L. Eye -> L. Ear (Violet)
]


# ============================================================================
# AUDIO FEATURE EXTRACTION
# ============================================================================

class SCAILAudioFeatureExtractor:
    """Extract audio features for pose animation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 10000}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "bass_range": ("STRING", {"default": "20-250", "tooltip": "Bass frequency range in Hz"}),
                "mid_range": ("STRING", {"default": "250-2000", "tooltip": "Mid frequency range in Hz"}),
                "treble_range": ("STRING", {"default": "2000-8000", "tooltip": "Treble frequency range in Hz"}),
                "smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_AUDIO_FEATURES",)
    RETURN_NAMES = ("audio_features",)
    FUNCTION = "extract"
    CATEGORY = "SCAIL-AudioReactive"
    
    def extract(self, audio, frame_count, fps, bass_range, mid_range, treble_range, smoothing):
        # Parse frequency ranges
        def parse_range(s):
            low, high = s.split("-")
            return int(low), int(high)
        
        bass_low, bass_high = parse_range(bass_range)
        mid_low, mid_high = parse_range(mid_range)
        treble_low, treble_high = parse_range(treble_range)
        
        # Get audio data
        waveform = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3:
            waveform = waveform[0]  # Remove batch dim
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  # Mono
        else:
            waveform = waveform[0]
        
        waveform = waveform.numpy()
        
        # Calculate samples per frame
        frame_duration = 1.0 / fps
        samples_per_frame = int(sample_rate * frame_duration)
        
        # Initialize feature arrays
        rms = np.zeros(frame_count)
        bass = np.zeros(frame_count)
        mid = np.zeros(frame_count)
        treble = np.zeros(frame_count)
        onsets = np.zeros(frame_count)
        
        # Process each frame
        for i in range(frame_count):
            start_sample = int(i * samples_per_frame)
            end_sample = min(start_sample + samples_per_frame, len(waveform))
            
            if start_sample >= len(waveform):
                break
                
            frame_audio = waveform[start_sample:end_sample]
            
            if len(frame_audio) == 0:
                continue
            
            # RMS energy
            rms[i] = np.sqrt(np.mean(frame_audio ** 2))
            
            # FFT for frequency bands
            if len(frame_audio) > 1:
                fft = np.fft.rfft(frame_audio)
                freqs = np.fft.rfftfreq(len(frame_audio), 1.0 / sample_rate)
                magnitudes = np.abs(fft)
                
                # Extract frequency bands
                bass_mask = (freqs >= bass_low) & (freqs < bass_high)
                mid_mask = (freqs >= mid_low) & (freqs < mid_high)
                treble_mask = (freqs >= treble_low) & (freqs < treble_high)
                
                bass[i] = np.mean(magnitudes[bass_mask]) if np.any(bass_mask) else 0
                mid[i] = np.mean(magnitudes[mid_mask]) if np.any(mid_mask) else 0
                treble[i] = np.mean(magnitudes[treble_mask]) if np.any(treble_mask) else 0
        
        # Onset detection (energy derivative)
        rms_diff = np.diff(rms, prepend=rms[0])
        onsets = np.maximum(rms_diff, 0)
        
        # Normalize all features to 0-1
        def normalize(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min > 1e-8:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.zeros_like(arr)
        
        rms = normalize(rms)
        bass = normalize(bass)
        mid = normalize(mid)
        treble = normalize(treble)
        onsets = normalize(onsets)
        
        # Apply smoothing (exponential moving average)
        if smoothing > 0:
            alpha = 1.0 - smoothing
            for arr in [rms, bass, mid, treble]:
                for i in range(1, len(arr)):
                    arr[i] = alpha * arr[i] + (1 - alpha) * arr[i-1]
        
        features = {
            "rms": torch.from_numpy(rms.copy()).float(),
            "bass": torch.from_numpy(bass.copy()).float(),
            "mid": torch.from_numpy(mid.copy()).float(),
            "treble": torch.from_numpy(treble.copy()).float(),
            "onsets": torch.from_numpy(onsets.copy()).float(),
            "frame_count": frame_count,
            "fps": fps,
        }
        
        return (features,)


# ============================================================================
# BASE POSE GENERATION
# ============================================================================

class SCAILBasePoseGenerator:
    """Generate a base pose (T-pose, A-pose, or custom)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_type": (["t_pose", "a_pose", "relaxed", "custom"], {"default": "relaxed"}),
                "height": ("FLOAT", {"default": 400.0, "min": 100.0, "max": 1000.0, "tooltip": "Character height in 3D units"}),
                "depth": ("FLOAT", {"default": 800.0, "min": 200.0, "max": 2000.0, "tooltip": "Distance from camera (Z)"}),
                "center_x": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0}),
                "center_y": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0}),
            },
            "optional": {
                "custom_pose": ("SCAIL_POSE",),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE",)
    RETURN_NAMES = ("base_pose",)
    FUNCTION = "generate"
    CATEGORY = "SCAIL-AudioReactive"
    
    def generate(self, pose_type, height, depth, center_x, center_y, custom_pose=None):
        if pose_type == "custom" and custom_pose is not None:
            return (custom_pose,)
        
        # Scale factor based on height
        scale = height / 400.0
        
        # Base skeleton proportions (relative to height=400)
        # All coordinates: X=left/right, Y=up/down, Z=depth
        base_joints = {
            "neck": [0, -120, 0],
            "nose": [0, -160, -20],
            "r_shoulder": [-60, -110, 0],
            "l_shoulder": [60, -110, 0],
            "r_hip": [-35, 40, 0],
            "l_hip": [35, 40, 0],
            "r_eye": [-15, -170, -25],
            "l_eye": [15, -170, -25],
            "r_ear": [-35, -160, 5],
            "l_ear": [35, -160, 5],
        }
        
        if pose_type == "t_pose":
            # Arms straight out
            base_joints.update({
                "r_elbow": [-130, -110, 0],
                "r_wrist": [-200, -110, 0],
                "l_elbow": [130, -110, 0],
                "l_wrist": [200, -110, 0],
                "r_knee": [-40, 120, 0],
                "r_ankle": [-40, 200, 0],
                "l_knee": [40, 120, 0],
                "l_ankle": [40, 200, 0],
            })
        elif pose_type == "a_pose":
            # Arms at ~45 degrees
            base_joints.update({
                "r_elbow": [-110, -60, 0],
                "r_wrist": [-160, -10, 0],
                "l_elbow": [110, -60, 0],
                "l_wrist": [160, -10, 0],
                "r_knee": [-40, 120, 0],
                "r_ankle": [-40, 200, 0],
                "l_knee": [40, 120, 0],
                "l_ankle": [40, 200, 0],
            })
        else:  # relaxed
            # Arms down, slight bend
            base_joints.update({
                "r_elbow": [-70, -20, 10],
                "r_wrist": [-80, 60, 20],
                "l_elbow": [70, -20, 10],
                "l_wrist": [80, 60, 20],
                "r_knee": [-45, 110, 10],
                "r_ankle": [-40, 195, 0],
                "l_knee": [45, 110, 10],
                "l_ankle": [40, 195, 0],
            })
        
        # Build joint array in COCO order
        joints = np.zeros((18, 3), dtype=np.float32)
        joint_map = {name: i for i, name in enumerate(JOINT_NAMES)}
        
        for name, coords in base_joints.items():
            idx = joint_map[name]
            joints[idx] = coords
        
        # Apply scale and position
        joints *= scale
        joints[:, 0] += center_x
        joints[:, 1] += center_y
        joints[:, 2] += depth
        
        pose = {
            "joints": torch.from_numpy(joints).float(),
            "joint_names": JOINT_NAMES,
            "limb_seq": LIMB_SEQ,
        }
        
        return (pose,)


# ============================================================================
# AUDIO REACTIVE POSE MODULATOR
# ============================================================================

class SCAILBeatDetector:
    """
    Detect beats, downbeats, and musical structure from audio.
    This is the foundation for beat-aligned dance generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 10000,
                    "tooltip": "Number of frames to generate (should match your video length)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_BEAT_INFO",)
    RETURN_NAMES = ("beat_info",)
    FUNCTION = "detect"
    CATEGORY = "SCAIL-AudioReactive"
    
    def detect(self, audio, frame_count, fps):
        import librosa
        
        # Extract audio data
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0) if waveform.shape[0] <= 2 else waveform[0]
        
        waveform = waveform.flatten().astype(np.float32)
        
        # Calculate duration from frame_count (user-specified)
        duration = frame_count / fps
        audio_duration = len(waveform) / sample_rate
        
        print(f"[SCAIL-BeatDetector] Target: {frame_count} frames ({duration:.2f}s), Audio: {audio_duration:.2f}s")
        
        # Beat tracking (on full audio)
        tempo, beat_frames_audio = librosa.beat.beat_track(y=waveform, sr=sample_rate)
        beat_times = librosa.frames_to_time(beat_frames_audio, sr=sample_rate)
        
        # Convert beat times to video frame indices, filter to frame_count
        beat_frames = (beat_times * fps).astype(int)
        beat_frames = beat_frames[beat_frames < frame_count]
        
        # Onset strength for energy detection
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        onset_times = librosa.times_like(onset_env, sr=sample_rate)
        
        # Resample onset envelope to target frame_count
        onset_env_resampled = np.interp(
            np.linspace(0, duration, frame_count),
            onset_times,
            onset_env
        )
        onset_env_resampled = onset_env_resampled / (onset_env_resampled.max() + 1e-8)
        
        # RMS energy per frame
        hop_length = 512
        rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
        rms_times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_length)
        rms_resampled = np.interp(
            np.linspace(0, duration, frame_count),
            rms_times,
            rms
        )
        rms_resampled = rms_resampled / (rms_resampled.max() + 1e-8)
        
        # Detect downbeats (every 4th beat typically)
        downbeat_frames = beat_frames[::4] if len(beat_frames) >= 4 else beat_frames
        
        # Create beat mask (1 on beat frames, 0 elsewhere)
        beat_mask = np.zeros(frame_count, dtype=np.float32)
        for bf in beat_frames:
            if bf < frame_count:
                beat_mask[bf] = 1.0
        
        downbeat_mask = np.zeros(frame_count, dtype=np.float32)
        for df in downbeat_frames:
            if df < frame_count:
                downbeat_mask[df] = 1.0
        
        # Estimate tempo as float
        tempo_float = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        print(f"[SCAIL-BeatDetector] Detected {len(beat_frames)} beats at {tempo_float:.1f} BPM")
        print(f"[SCAIL-BeatDetector] {len(downbeat_frames)} downbeats in {frame_count} frames")
        
        beat_info = {
            "frame_count": frame_count,
            "fps": fps,
            "tempo": tempo_float,
            "beat_frames": beat_frames,
            "downbeat_frames": downbeat_frames,
            "beat_mask": torch.from_numpy(beat_mask).float(),
            "downbeat_mask": torch.from_numpy(downbeat_mask).float(),
            "onset_envelope": torch.from_numpy(onset_env_resampled).float(),
            "energy": torch.from_numpy(rms_resampled).float(),
            "duration": duration,
        }
        
        return (beat_info,)


# ============================================================================
# JOINT ROTATION SYSTEM
# ============================================================================

# Joint chains for hierarchical rotation (parent affects children)
JOINT_CHAINS = {
    "spine": [1],  # neck is root
    "head": [1, 0, 14, 15, 16, 17],  # neck -> nose -> eyes -> ears
    "right_arm": [1, 2, 3, 4],  # neck -> shoulder -> elbow -> wrist
    "left_arm": [1, 5, 6, 7],
    "right_leg": [1, 8, 9, 10],  # neck -> hip -> knee -> ankle
    "left_leg": [1, 11, 12, 13],
}

# Rotation axes for each joint (which axis to rotate around for each motion type)
# Format: {"joint_idx": {"motion_name": (axis, angle_multiplier)}}
# Axis: 0=X (forward/back), 1=Y (turn left/right), 2=Z (tilt side)
JOINT_ROTATION_AXES = {
    # Head rotations (pivot at neck)
    0: {"nod": (0, 1.0), "turn": (1, 1.0), "tilt": (2, 1.0)},
    # Shoulder rotations
    2: {"raise": (2, -1.0), "forward": (0, 1.0)},  # right shoulder
    5: {"raise": (2, 1.0), "forward": (0, 1.0)},   # left shoulder
    # Elbow rotations
    3: {"bend": (0, 1.0), "twist": (1, 1.0)},  # right elbow
    6: {"bend": (0, 1.0), "twist": (1, 1.0)},  # left elbow
    # Hip rotations  
    8: {"raise": (2, -1.0), "forward": (0, 1.0)},  # right hip
    11: {"raise": (2, 1.0), "forward": (0, 1.0)},  # left hip
    # Knee rotations
    9: {"bend": (0, 1.0)},   # right knee 
    12: {"bend": (0, 1.0)},  # left knee
    # Ankle rotations - to keep feet flat when knees bend
    10: {"bend": (0, -1.0)},  # right ankle (opposite to knee)
    13: {"bend": (0, -1.0)},  # left ankle
}


def rotate_point_around_pivot(point: np.ndarray, pivot: np.ndarray, 
                               axis: int, angle: float) -> np.ndarray:
    """Rotate a point around a pivot on a given axis.
    
    Args:
        point: 3D point to rotate
        pivot: 3D pivot point
        axis: 0=X, 1=Y, 2=Z
        angle: rotation angle in radians
    """
    # Translate to origin
    p = point - pivot
    
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 0:  # X-axis rotation (pitch/nod)
        rotated = np.array([
            p[0],
            p[1] * c - p[2] * s,
            p[1] * s + p[2] * c
        ])
    elif axis == 1:  # Y-axis rotation (yaw/turn)
        rotated = np.array([
            p[0] * c + p[2] * s,
            p[1],
            -p[0] * s + p[2] * c
        ])
    else:  # Z-axis rotation (roll/tilt)
        rotated = np.array([
            p[0] * c - p[1] * s,
            p[0] * s + p[1] * c,
            p[2]
        ])
    
    return rotated + pivot


def apply_rotation_to_chain(joints: np.ndarray, pivot_idx: int, 
                            affected_indices: List[int], axis: int, 
                            angle: float) -> np.ndarray:
    """Apply rotation to a chain of joints.
    
    Args:
        joints: (18, 3) joint positions
        pivot_idx: index of pivot joint
        affected_indices: list of joint indices to rotate (excluding pivot)
        axis: rotation axis
        angle: rotation angle in radians
    """
    result = joints.copy()
    pivot = joints[pivot_idx]
    
    for idx in affected_indices:
        if idx != pivot_idx:
            result[idx] = rotate_point_around_pivot(joints[idx], pivot, axis, angle)
    
    return result


# ============================================================================
# POSE LIBRARY - Now using rotations instead of raw offsets
# ============================================================================

# Each pose defines rotations for body parts
# Format: {"joint_group": {"motion": angle_in_degrees}}
DANCE_POSES = {
    "neutral": {
        "description": "Relaxed standing pose",
        "rotations": {}
    },
    "arms_up": {
        "description": "Both arms raised",
        "rotations": {
            "right_shoulder": {"raise": 120},
            "left_shoulder": {"raise": 120},
            "right_elbow": {"bend": -30},
            "left_elbow": {"bend": -30},
        }
    },
    "arms_out": {
        "description": "Arms spread wide (T-pose style)",
        "rotations": {
            "right_shoulder": {"raise": 80},
            "left_shoulder": {"raise": 80},
        }
    },
    "lean_left": {
        "description": "Body leaning left",
        "rotations": {
            "spine": {"tilt": 15},
            "head": {"tilt": -5},
        },
        "translations": {
            "upper": {"x": -6},
        }
    },
    "lean_right": {
        "description": "Body leaning right",
        "rotations": {
            "spine": {"tilt": -15},
            "head": {"tilt": 5},
        },
        "translations": {
            "upper": {"x": 6},
        }
    },
    "crouch": {
        "description": "Knees bent, lower stance",
        "rotations": {
            "right_knee": {"bend": 40},
            "left_knee": {"bend": 40},
            "right_ankle": {"bend": 20},
            "left_ankle": {"bend": 20},
            "spine": {"forward": 8},
        },
        "translations": {
            "all": {"y": 15},
        }
    },
    "pump_right": {
        "description": "Right fist pump upward",
        "rotations": {
            "right_shoulder": {"raise": 80, "forward": 10},
            "right_elbow": {"bend": 110},
        }
    },
    "pump_left": {
        "description": "Left fist pump upward",
        "rotations": {
            "left_shoulder": {"raise": 80, "forward": 10},
            "left_elbow": {"bend": 110},
        }
    },
    "head_down": {
        "description": "Head nodded down",
        "rotations": {
            "head": {"nod": 25},
        }
    },
    "head_back": {
        "description": "Head tilted back",
        "rotations": {
            "head": {"nod": -20},
        }
    },
    "hip_left": {
        "description": "Hips shifted left",
        "rotations": {
            "spine": {"tilt": 12},
            "left_knee": {"bend": 25},
            "right_knee": {"bend": 8},
            "left_ankle": {"bend": 12},
            "right_ankle": {"bend": 4},
        },
        "translations": {
            "hips": {"x": -8},  # Subtle shift
        }
    },
    "hip_right": {
        "description": "Hips shifted right",
        "rotations": {
            "spine": {"tilt": -12},
            "right_knee": {"bend": 25},
            "left_knee": {"bend": 8},
            "right_ankle": {"bend": 12},
            "left_ankle": {"bend": 4},
        },
        "translations": {
            "hips": {"x": 8},  # Subtle shift
        }
    },
    "slide_right": {
        "description": "Weight slides right, both feet grounded",
        "rotations": {
            "spine": {"tilt": -15},
            "right_knee": {"bend": 35},
            "left_knee": {"bend": 10},
            "right_ankle": {"bend": 18},
            "left_ankle": {"bend": 5},
        },
        "translations": {
            "hips": {"x": 10},
        }
    },
    "slide_left": {
        "description": "Weight slides left, both feet grounded",
        "rotations": {
            "spine": {"tilt": 15},
            "left_knee": {"bend": 35},
            "right_knee": {"bend": 10},
            "left_ankle": {"bend": 18},
            "right_ankle": {"bend": 5},
        },
        "translations": {
            "hips": {"x": -10},
        }
    },
    "shuffle_right": {
        "description": "Shuffle step right, feet grounded",
        "rotations": {
            "spine": {"tilt": -12, "turn": -15},
            "right_knee": {"bend": 30},
            "left_knee": {"bend": 15},
            "right_ankle": {"bend": 15},
            "left_ankle": {"bend": 8},
        },
        "translations": {
            "hips": {"x": 8},
        }
    },
    "shuffle_left": {
        "description": "Shuffle step left, feet grounded",
        "rotations": {
            "spine": {"tilt": 12, "turn": 15},
            "left_knee": {"bend": 30},
            "right_knee": {"bend": 15},
            "left_ankle": {"bend": 15},
            "right_ankle": {"bend": 8},
        },
        "translations": {
            "hips": {"x": -8},
        }
    },
    "groove_right": {
        "description": "Groove with weight on right",
        "rotations": {
            "right_knee": {"bend": 40},
            "left_knee": {"bend": 15},
            "right_ankle": {"bend": 20},
            "left_ankle": {"bend": 8},
            "spine": {"tilt": -8, "forward": 8},
        },
        "translations": {
            "hips": {"x": 6, "y": 5},
        }
    },
    "groove_left": {
        "description": "Groove with weight on left",
        "rotations": {
            "left_knee": {"bend": 40},
            "right_knee": {"bend": 15},
            "left_ankle": {"bend": 20},
            "right_ankle": {"bend": 8},
            "spine": {"tilt": 8, "forward": 8},
        },
        "translations": {
            "hips": {"x": -6, "y": 5},
        }
    },
    "arms_crossed_low": {
        "description": "Arms crossed in front, low",
        "rotations": {
            "right_shoulder": {"forward": 40, "raise": 20},
            "left_shoulder": {"forward": 40, "raise": 20},
            "right_elbow": {"bend": 60},
            "left_elbow": {"bend": 60},
        }
    },
    "look_left": {
        "description": "Head turned left",
        "rotations": {
            "head": {"turn": 30},
        }
    },
    "look_right": {
        "description": "Head turned right",
        "rotations": {
            "head": {"turn": -30},
        }
    },
}

# Move sequences for different energy levels
# Natural flow - no squatting between every move!
MOVE_SEQUENCES = {
    "low": [
        # Gentle swaying
        ["lean_left", "lean_right", "lean_left", "lean_right"],
        ["hip_left", "hip_right", "hip_left", "hip_right"],
        ["look_left", "look_right", "look_left", "look_right"],
        # Subtle grooving
        ["groove_left", "groove_right", "groove_left", "groove_right"],
        ["slide_left", "slide_right", "slide_left", "slide_right"],
        ["hip_left", "lean_left", "hip_right", "lean_right"],
    ],
    "medium": [
        # Arm flows - let arms actually move!
        ["pump_right", "pump_left", "pump_right", "pump_left"],
        ["arms_out", "arms_up", "arms_out", "arms_up"],
        ["pump_right", "arms_up", "pump_left", "arms_up"],
        ["arms_up", "arms_out", "arms_up", "arms_crossed_low"],
        # Body + arms together
        ["lean_left", "pump_left", "lean_right", "pump_right"],
        ["hip_left", "arms_up", "hip_right", "arms_up"],
        ["groove_left", "pump_left", "groove_right", "pump_right"],
        # Slides with arms
        ["slide_left", "arms_out", "slide_right", "arms_out"],
        ["shuffle_left", "pump_left", "shuffle_right", "pump_right"],
        ["groove_left", "arms_up", "groove_right", "arms_up"],
    ],
    "high": [
        # High energy arms
        ["arms_up", "pump_right", "arms_up", "pump_left"],
        ["pump_right", "arms_up", "pump_left", "arms_up"],
        ["arms_out", "arms_up", "arms_out", "arms_up"],
        # Movement + arms
        ["slide_right", "arms_up", "slide_left", "arms_up"],
        ["shuffle_left", "arms_up", "shuffle_right", "arms_up"],
        ["groove_right", "pump_right", "groove_left", "pump_left"],
        # Full combos - crouch only as accent
        ["groove_left", "arms_up", "lean_right", "arms_out"],
        ["slide_left", "pump_left", "slide_right", "pump_right"],
        ["hip_left", "arms_up", "hip_right", "pump_right", "hip_left", "pump_left"],
        # Crouch as rare accent, not constant
        ["arms_up", "crouch", "arms_up", "arms_out"],
        ["pump_right", "pump_left", "crouch", "arms_up"],
    ],
}


# ============================================================================
# AUDIO FEATURE MAPPING
# ============================================================================

# Which audio features drive which body parts
AUDIO_FEATURE_MAPPING = {
    "bass": {
        "description": "Low frequencies - drives bounce and knee bend",
        "affects": {
            "bounce": 1.0,           # Global vertical motion
            "right_knee": {"bend": 0.6},
            "left_knee": {"bend": 0.6},
            "right_hip": {"forward": 0.3},
            "left_hip": {"forward": 0.3},
        }
    },
    "mid": {
        "description": "Mid frequencies - drives sway and torso",
        "affects": {
            "sway": 1.0,             # Global horizontal motion
            "spine": {"tilt": 0.5},
            "hip_sway": 0.4,
        }
    },
    "treble": {
        "description": "High frequencies - drives arms and hands",
        "affects": {
            "right_shoulder": {"raise": 0.4},
            "left_shoulder": {"raise": 0.4},
            "right_elbow": {"bend": 0.3},
            "left_elbow": {"bend": 0.3},
        }
    },
    "onset": {
        "description": "Transients/hits - drives head snaps and accents",
        "affects": {
            "head": {"nod": 0.8},
            "accent_scale": 1.5,     # Momentary intensity boost
        }
    },
    "energy": {
        "description": "Overall RMS energy - scales everything",
        "affects": {
            "global_scale": 1.0,
        }
    },
}


# ============================================================================
# MOTION DYNAMICS SYSTEM
# ============================================================================
# Proper animation principles: momentum, follow-through, overlapping action

class MotionDynamics:
    """
    Physics-inspired motion system for fluid animation.
    
    Instead of linear interpolation between keyframes, this system:
    1. Tracks velocity per joint
    2. Applies momentum and drag
    3. Uses different timing for different body parts (overlapping action)
    4. Adds follow-through after reaching targets
    """
    
    # Body part groupings for overlapping action
    # Lower numbers = leads the motion, higher = follows/drags
    BODY_PART_PRIORITY = {
        # Core leads
        1: [1],  # neck/spine - initiates
        8: [8], 11: [11],  # hips - power center
        
        # Mid-chain follows
        2: [2], 5: [5],  # shoulders
        9: [9], 12: [12],  # knees
        
        # Extremities drag behind
        0: [0, 14, 15, 16, 17],  # head/face
        3: [3], 6: [6],  # elbows
        10: [10], 13: [13],  # ankles
        4: [4], 7: [7],  # wrists (most drag)
    }
    
    # Drag factors per joint (0 = instant, higher = more follow-through)
    # Keep low for responsive motion
    JOINT_DRAG = {
        0: 0.10,  # nose
        1: 0.03,  # neck - leads
        2: 0.08,  # r_shoulder - responsive
        3: 0.12,  # r_elbow
        4: 0.18,  # r_wrist - trails
        5: 0.08,  # l_shoulder
        6: 0.12,  # l_elbow
        7: 0.18,  # l_wrist
        8: 0.01,  # r_hip (snapped anyway)
        9: 0.01,  # r_knee
        10: 0.01, # r_ankle
        11: 0.01, # l_hip
        12: 0.01, # l_knee
        13: 0.01, # l_ankle
        14: 0.10, # r_eye
        15: 0.10, # l_eye
        16: 0.15, # r_ear
        17: 0.15, # l_ear
    }
    
    # Stiffness - MUCH higher for actual visible motion
    JOINT_STIFFNESS = {
        0: 25.0,  # nose
        1: 35.0,  # neck - snappy
        2: 30.0,  # shoulders - responsive!
        3: 25.0,  # elbows
        4: 20.0,  # wrists
        5: 30.0,
        6: 25.0,
        7: 20.0,
        8: 80.0,  # hips (snapped anyway)
        9: 80.0,  # knees
        10: 80.0, # ankles
        11: 80.0,
        12: 80.0,
        13: 80.0,
        14: 20.0, # eyes
        15: 20.0,
        16: 15.0, # ears
        17: 15.0,
    }
    
    def __init__(self, num_joints=18):
        self.num_joints = num_joints
        self.velocities = np.zeros((num_joints, 3))
        self.positions = None
        self.dt = 1.0 / 24.0  # Will be set per frame
        
    def set_fps(self, fps):
        self.dt = 1.0 / fps
        
    def initialize(self, start_pose):
        """Initialize with starting pose."""
        self.positions = start_pose.copy()
        self.velocities = np.zeros((self.num_joints, 3))
    
    def step_towards(self, target_pose, urgency=1.0):
        """
        Step the motion system towards a target pose.
        
        Args:
            target_pose: (18, 3) target joint positions
            urgency: 0-2, how quickly to move (1 = normal, 2 = snappy, 0.5 = lazy)
        
        Returns:
            Current pose after physics step
        """
        if self.positions is None:
            self.positions = target_pose.copy()
            return self.positions.copy()
        
        # Leg joints that should SNAP not blend (to prevent both-legs-up)
        LEG_JOINTS = {8, 9, 10, 11, 12, 13}  # hips, knees, ankles
        
        result = np.zeros_like(self.positions)
        
        for j in range(self.num_joints):
            # LEGS: Snap directly toward target (no smooth physics)
            # This prevents both legs being up during transitions
            if j in LEG_JOINTS:
                # Fast lerp instead of spring physics
                snap_speed = 0.4 * urgency  # 40% toward target per frame
                self.positions[j] = self.positions[j] * (1 - snap_speed) + target_pose[j] * snap_speed
                self.velocities[j] = np.zeros(3)  # No momentum for legs
                result[j] = self.positions[j]
                continue
            
            # EVERYTHING ELSE: Spring-damper physics for fluid motion
            drag = self.JOINT_DRAG.get(j, 0.10)
            stiffness = self.JOINT_STIFFNESS.get(j, 25.0) * urgency
            
            # Spring force towards target
            displacement = target_pose[j] - self.positions[j]
            spring_force = displacement * stiffness
            
            # Damping - lower value = more lively, allows overshoot
            damping = 1.0 * np.sqrt(stiffness)
            damping_force = -self.velocities[j] * damping
            
            # Total acceleration
            acceleration = spring_force + damping_force
            
            # Integrate velocity
            self.velocities[j] += acceleration * self.dt
            
            # Apply drag to velocity (separate from spring damping)
            self.velocities[j] *= (1.0 - drag)
            self.velocities[j] *= (1.0 - drag * 0.5)
            
            # Integrate position
            self.positions[j] += self.velocities[j] * self.dt
            
            result[j] = self.positions[j]
        
        return result
    
    def add_impulse(self, joint_indices, impulse_vector):
        """Add an instantaneous velocity impulse to joints."""
        for j in joint_indices:
            if j < self.num_joints:
                self.velocities[j] += impulse_vector
    
    def get_velocity_magnitude(self):
        """Get overall motion intensity."""
        return np.mean(np.linalg.norm(self.velocities, axis=1))


def ease_in_out_cubic(t):
    """Smooth ease-in-out curve."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_back(t, overshoot=0.3):
    """Ease out with slight overshoot (anticipation/follow-through)."""
    c1 = 1.70158 * overshoot
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def ease_out_elastic(t, intensity=0.3):
    """Elastic ease out - for snappy motions with bounce."""
    if t == 0 or t == 1:
        return t
    p = 0.3
    return pow(2, -10 * t) * np.sin((t - p / 4) * (2 * np.pi) / p) * intensity + 1


class SCAILBeatDrivenPose:
    """
    Beat-driven keyframe dance system with proper joint rotation and audio modulation.
    
    This system:
    1. Detects beats from audio
    2. Selects poses from a library on each beat
    3. Uses physics-based motion dynamics (momentum, follow-through, drag)
    4. Overlapping action - different body parts move at different rates
    5. Overlays continuous audio-driven modulation (bass→bounce, treble→arms, etc.)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pose": ("SCAIL_POSE",),
                "beat_info": ("SCAIL_BEAT_INFO",),
                "audio_features": ("SCAIL_AUDIO_FEATURES",),
                
                # Style/Intensity
                "energy_style": (["auto", "low", "medium", "high"], {"default": "auto",
                    "tooltip": "Dance energy level - auto detects from audio"}),
                "pose_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Scale keyframe pose rotations"}),
                
                # Motion dynamics
                "motion_smoothness": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 2.0, "step": 0.1,
                    "tooltip": "How smooth/fluid the motion is (higher = more follow-through)"}),
                "motion_snap": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "How snappy motion is on beats (higher = sharper hits)"}),
                
                # Audio-driven modulation intensities
                "bass_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Bass → bounce, knee bend"}),
                "mid_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Mid → sway, torso lean"}),
                "treble_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Treble → arm raise, hand energy"}),
                "onset_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Onset → head snap, accents"}),
                
                # Variation
                "pose_variation": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Random variation in pose selection"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("pose_sequence",)
    FUNCTION = "generate"
    CATEGORY = "SCAIL-AudioReactive"
    
    # Joint group to joint indices mapping
    JOINT_GROUPS = {
        "head": [0, 14, 15, 16, 17],
        "spine": [1, 2, 5, 8, 11],  # neck affects shoulders and hips
        "right_shoulder": [2, 3, 4],
        "left_shoulder": [5, 6, 7],
        "right_elbow": [3, 4],
        "left_elbow": [6, 7],
        "right_hip": [8, 9, 10],
        "left_hip": [11, 12, 13],
        "right_knee": [9, 10],
        "left_knee": [12, 13],
        "right_ankle": [10],
        "left_ankle": [13],
    }
    
    # Pivot joints for each group
    PIVOT_JOINTS = {
        "head": 1,      # neck
        "spine": 1,     # neck (simplified - ideally would be pelvis)
        "right_shoulder": 2,
        "left_shoulder": 5,
        "right_elbow": 3,
        "left_elbow": 6,
        "right_hip": 8,
        "left_hip": 11,
        "right_knee": 9,
        "left_knee": 12,
        "right_ankle": 10,
        "left_ankle": 13,
    }
    
    # Rotation axis for each motion type
    MOTION_AXES = {
        "nod": 0,      # X - forward/back tilt
        "turn": 1,     # Y - left/right rotation
        "tilt": 2,     # Z - side tilt
        "raise": 2,    # Z - for arms/legs lifting sideways
        "forward": 0,  # X - for arms/legs moving forward
        "bend": 0,     # X - for elbow/knee bending
        "twist": 1,    # Y - for arm twist
    }
    
    def _apply_rotation(self, joints: np.ndarray, group: str, motion: str, 
                        angle_deg: float) -> np.ndarray:
        """Apply a rotation to a joint group."""
        if group not in self.JOINT_GROUPS or group not in self.PIVOT_JOINTS:
            return joints
        
        if motion not in self.MOTION_AXES:
            return joints
        
        affected = self.JOINT_GROUPS[group]
        pivot_idx = self.PIVOT_JOINTS[group]
        axis = self.MOTION_AXES[motion]
        angle_rad = np.radians(angle_deg)
        
        # Handle sign adjustments for left/right symmetry
        if "left" in group and motion in ["raise", "tilt"]:
            angle_rad = -angle_rad
        
        return apply_rotation_to_chain(joints, pivot_idx, affected, axis, angle_rad)
    
    def _apply_pose(self, base_joints: np.ndarray, pose_name: str, 
                    intensity: float = 1.0) -> np.ndarray:
        """Apply a named pose using rotations and translations."""
        if pose_name not in DANCE_POSES:
            return base_joints.copy()
        
        pose_def = DANCE_POSES[pose_name]
        rotations = pose_def.get("rotations", {})
        translations = pose_def.get("translations", {})
        
        result = base_joints.copy()
        
        # Apply rotations
        for group, motions in rotations.items():
            for motion, angle in motions.items():
                result = self._apply_rotation(result, group, motion, angle * intensity)
        
        # Apply translations (direct joint position offsets)
        # "all" moves all joints, or specify joint indices
        for target, offset in translations.items():
            if target == "all":
                for i in range(len(result)):
                    result[i, 0] += offset.get("x", 0) * intensity
                    result[i, 1] += offset.get("y", 0) * intensity
                    result[i, 2] += offset.get("z", 0) * intensity
            elif target == "upper":  # Everything except legs
                for i in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]:
                    result[i, 0] += offset.get("x", 0) * intensity
                    result[i, 1] += offset.get("y", 0) * intensity
                    result[i, 2] += offset.get("z", 0) * intensity
            elif target == "hips":  # Core body shift
                for i in [1, 2, 5, 8, 11]:  # neck, shoulders, hips
                    result[i, 0] += offset.get("x", 0) * intensity
                    result[i, 1] += offset.get("y", 0) * intensity
                    result[i, 2] += offset.get("z", 0) * intensity
        
        return result
    
    def _interpolate_poses(self, pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
        """Smoothly interpolate between two poses using ease-in-out."""
        if t < 0.5:
            t_smooth = 4 * t * t * t
        else:
            t_smooth = 1 - pow(-2 * t + 2, 3) / 2
        
        return pose_a * (1 - t_smooth) + pose_b * t_smooth
    
    def _select_move_sequence(self, energy_level: str, rng: np.random.Generator, 
                               variation: float) -> List[str]:
        """Select a move sequence based on energy level."""
        sequences = MOVE_SEQUENCES.get(energy_level, MOVE_SEQUENCES["medium"])
        
        if variation > 0 and rng.random() < variation:
            levels = ["low", "medium", "high"]
            current_idx = levels.index(energy_level)
            if rng.random() < 0.5 and current_idx > 0:
                sequences = MOVE_SEQUENCES[levels[current_idx - 1]]
            elif current_idx < 2:
                sequences = MOVE_SEQUENCES[levels[current_idx + 1]]
        
        return sequences[rng.integers(0, len(sequences))]
    
    def _determine_energy_level(self, energy: np.ndarray, frame: int) -> str:
        """Determine energy level at a given frame."""
        if frame >= len(energy):
            return "medium"
        
        start = max(0, frame - 5)
        end = min(len(energy), frame + 5)
        local_energy = np.mean(energy[start:end])
        
        if local_energy < 0.33:
            return "low"
        elif local_energy < 0.66:
            return "medium"
        else:
            return "high"
    
    def _apply_audio_modulation(self, joints: np.ndarray, base_joints: np.ndarray,
                                 bass: float, mid: float, treble: float, onset: float,
                                 bass_int: float, mid_int: float, treble_int: float, 
                                 onset_int: float, time_sec: float, tempo: float) -> np.ndarray:
        """Apply continuous audio-driven modulation as an overlay.
        
        This is additive motion on top of the keyframe poses - subtle, continuous,
        and reactive to audio without fighting the base motion.
        """
        result = joints.copy()
        
        beat_period = 60.0 / tempo
        phase = (time_sec / beat_period) * 2 * np.pi
        
        # Half-time phase (bounce every other beat - more natural)
        half_phase = phase * 0.5
        
        # === BASS: Down-on-the-beat bounce ===
        if bass_int > 0:
            # Sharp down, smooth up - like a real bounce
            # Use a curve that goes down quickly on beat, rises slowly between
            # cos gives us: 1 at phase=0 (between beats), -1 at phase=pi (on beat)
            # We want DOWN on beat, so positive values = down
            bounce_curve = -np.cos(half_phase)  # -1 to 1, peaks DOWN at downbeat
            
            # Make it punchier: sharpen the curve so down is quick, up is gradual
            if bounce_curve > 0:
                bounce_curve = bounce_curve ** 0.7  # Sharpen the down
            else:
                bounce_curve = -((-bounce_curve) ** 1.5)  # Soften the up
            
            bounce = bass * bass_int * 6.0 * max(0, bounce_curve)  # Subtle bounce
            result[:, 1] += bounce
            
            # Subtle knee pulse - not a squat!
            knee_bend = bass * bass_int * 6 * max(0, bounce_curve)
            ankle_bend = knee_bend * 0.5
            result = self._apply_rotation(result, "right_knee", "bend", knee_bend)
            result = self._apply_rotation(result, "left_knee", "bend", knee_bend)
            result = self._apply_rotation(result, "right_ankle", "bend", ankle_bend)
            result = self._apply_rotation(result, "left_ankle", "bend", ankle_bend)
        
        # === MID: Sway (half-time, alternating sides) ===
        if mid_int > 0:
            # Smooth horizontal sway at half tempo
            sway = np.sin(half_phase) * mid * mid_int * 10.0
            result[:, 0] += sway
            
            # Subtle spine follow
            result = self._apply_rotation(result, "spine", "tilt", 
                                          np.sin(half_phase) * mid * mid_int * 4)
        
        # === TREBLE: Arm energy ===
        if treble_int > 0:
            # Arms lift slightly with treble
            arm_lift = treble * treble_int * 12
            result = self._apply_rotation(result, "right_shoulder", "raise", arm_lift)
            result = self._apply_rotation(result, "left_shoulder", "raise", arm_lift)
            
            # Elbow variation at half-time
            elbow_var = 20 + np.sin(half_phase) * treble * treble_int * 10
            result = self._apply_rotation(result, "right_elbow", "bend", elbow_var)
            result = self._apply_rotation(result, "left_elbow", "bend", elbow_var)
        
        # === ONSET: Head accent ===
        if onset_int > 0 and onset > 0.4:
            head_snap = onset * onset_int * 12
            result = self._apply_rotation(result, "head", "nod", head_snap)
        
        return result
    
    def generate(self, base_pose, beat_info, audio_features, energy_style, pose_intensity,
                 motion_smoothness, motion_snap, bass_intensity, mid_intensity,
                 treble_intensity, onset_intensity, pose_variation, seed):
        
        rng = np.random.default_rng(seed)
        
        base_joints = base_pose["joints"].numpy()  # (18, 3)
        frame_count = beat_info["frame_count"]
        fps = beat_info["fps"]
        tempo = beat_info["tempo"]
        beat_frames = beat_info["beat_frames"]
        downbeat_frames = beat_info["downbeat_frames"]
        energy = beat_info["energy"].numpy()
        
        # Get audio feature arrays
        bass = audio_features["bass"].numpy()
        mid = audio_features["mid"].numpy()
        treble = audio_features["treble"].numpy()
        onsets = audio_features["onsets"].numpy()
        
        # Ensure audio features match frame count
        if len(bass) != frame_count:
            bass = np.interp(np.linspace(0, 1, frame_count), 
                            np.linspace(0, 1, len(bass)), bass)
            mid = np.interp(np.linspace(0, 1, frame_count),
                           np.linspace(0, 1, len(mid)), mid)
            treble = np.interp(np.linspace(0, 1, frame_count),
                              np.linspace(0, 1, len(treble)), treble)
            onsets = np.interp(np.linspace(0, 1, frame_count),
                              np.linspace(0, 1, len(onsets)), onsets)
        
        print(f"[SCAIL-BeatDriven] Generating {frame_count} frames with {len(beat_frames)} beats")
        print(f"[SCAIL-BeatDriven] Tempo: {tempo:.1f} BPM, Motion dynamics enabled")
        print(f"[SCAIL-BeatDriven] Smoothness: {motion_smoothness}, Snap: {motion_snap}")
        
        # All poses are now grounded (slide/shuffle style) - no leg lifts
        GROUNDED_POSES = {"neutral", "crouch", "lean_left", "lean_right", "hip_left", "hip_right", 
                         "arms_up", "arms_out", "pump_left", "pump_right", "arms_crossed_low",
                         "head_down", "head_back", "look_left", "look_right",
                         "slide_left", "slide_right", "shuffle_left", "shuffle_right",
                         "groove_left", "groove_right"}
        
        # Build keyframe schedule
        keyframes = []
        current_sequence = []
        sequence_idx = 0
        last_pose = "neutral"
        
        for i, beat_frame in enumerate(beat_frames):
            is_downbeat = beat_frame in downbeat_frames
            beat_frame = int(beat_frame)
            
            if energy_style == "auto":
                energy_level = self._determine_energy_level(energy, beat_frame)
            else:
                energy_level = energy_style
            
            # Get next pose from sequence, or start new sequence
            if sequence_idx >= len(current_sequence):
                current_sequence = self._select_move_sequence(energy_level, rng, pose_variation)
                sequence_idx = 0
            
            pose_name = current_sequence[sequence_idx]
            sequence_idx += 1
            
            # On downbeats, maybe add emphasis
            if is_downbeat and rng.random() < 0.3:
                if last_pose in ["lean_left", "lean_right", "hip_left", "hip_right", 
                                "slide_left", "slide_right", "shuffle_left", "shuffle_right"]:
                    pose_name = "crouch"
                elif last_pose in ["pump_right", "pump_left"]:
                    pose_name = "arms_up"
                elif last_pose == "crouch":
                    pose_name = "arms_up"
            
            keyframes.append((beat_frame, pose_name, is_downbeat))
            last_pose = pose_name
        
        # Only add neutral at very start if needed (let it start moving immediately)
        if len(keyframes) == 0:
            keyframes.append((0, "neutral", False))
        elif keyframes[0][0] > 10:
            # Long gap at start - ease in from neutral
            keyframes.insert(0, (0, "neutral", False))
        
        # End: return to a relaxed pose
        if len(keyframes) > 0 and keyframes[-1][0] < frame_count - 10:
            final_pose = "neutral" if last_pose == "arms_up" else last_pose
            keyframes.append((frame_count - 1, final_pose, False))
        
        print(f"[SCAIL-BeatDriven] Created {len(keyframes)} keyframes")
        
        # Pre-compute keyframe poses using rotation system
        keyframe_poses = {}
        for frame_idx, pose_name, _ in keyframes:
            if frame_idx not in keyframe_poses:
                keyframe_poses[frame_idx] = self._apply_pose(
                    base_joints, pose_name, pose_intensity
                )
        
        # Initialize motion dynamics system
        motion = MotionDynamics(num_joints=18)
        motion.set_fps(fps)
        motion.initialize(base_joints.copy())
        
        # Apply smoothness parameter - subtle scaling, not extreme
        # smoothness > 1 = slightly more follow-through
        # smoothness < 1 = slightly snappier
        drag_scale = 0.7 + 0.3 * motion_smoothness  # 0.7 to 1.3 range
        stiffness_scale = 1.3 - 0.3 * motion_smoothness  # 1.0 to 0.7 range (inverted)
        
        for j in range(18):
            base_drag = MotionDynamics.JOINT_DRAG.get(j, 0.1)
            base_stiff = MotionDynamics.JOINT_STIFFNESS.get(j, 15.0)
            motion.JOINT_DRAG[j] = base_drag * drag_scale
            motion.JOINT_STIFFNESS[j] = base_stiff * stiffness_scale
        
        # Generate pose for each frame using physics simulation
        pose_sequence = []
        
        for frame_idx in range(frame_count):
            # Find current target keyframe
            target_kf = None
            next_kf = None
            
            for kf_frame, kf_pose, kf_downbeat in keyframes:
                if kf_frame <= frame_idx:
                    target_kf = (kf_frame, kf_pose, kf_downbeat)
                if kf_frame > frame_idx and next_kf is None:
                    next_kf = (kf_frame, kf_pose, kf_downbeat)
                    break
            
            if target_kf is None:
                target_kf = keyframes[0]
            
            target_frame, target_pose_name, is_downbeat = target_kf
            target_joints = keyframe_poses.get(target_frame, base_joints)
            
            # Calculate urgency based on proximity to next beat
            # More urgent = snappier motion as we approach the beat
            if next_kf is not None:
                next_frame = next_kf[0]
                next_pose_name = next_kf[1]
                frames_to_next = next_frame - frame_idx
                frames_since_last = frame_idx - target_frame
                total_span = next_frame - target_frame
                
                if total_span > 0:
                    progress = frames_since_last / total_span
                    
                    # Anticipation: start moving toward next pose early
                    if progress > 0.7:
                        # Blend target toward next pose
                        next_joints = keyframe_poses.get(next_kf[0], base_joints)
                        blend = (progress - 0.7) / 0.3  # 0 to 1 over last 30%
                        blend = ease_in_out_cubic(blend) * 0.4  # Max 40% blend
                        target_joints = target_joints * (1 - blend) + next_joints * blend
                    
                    # Urgency peaks mid-transition, lower at endpoints
                    base_urgency = 0.7 / motion_smoothness
                    urgency_variation = 0.5 * motion_snap
                    urgency = base_urgency + urgency_variation * np.sin(progress * np.pi)
                else:
                    urgency = 1.0 * motion_snap
            else:
                urgency = 0.7 / motion_smoothness  # Relaxed at end
            
            # Downbeats get extra snap
            if is_downbeat and frame_idx == target_frame:
                urgency *= 1.2 * motion_snap
            
            # Step physics simulation toward target
            joints = motion.step_towards(target_joints, urgency)
            
            # Apply audio-driven modulation as subtle overlay
            time_sec = frame_idx / fps
            joints = self._apply_audio_modulation(
                joints, base_joints,
                bass[frame_idx], mid[frame_idx], treble[frame_idx], onsets[frame_idx],
                bass_intensity, mid_intensity, treble_intensity, onset_intensity,
                time_sec, tempo
            )
            
            # Add impulse on strong onsets for extra pop
            if onsets[frame_idx] > 0.6 and onset_intensity > 0:
                # Small upward impulse to head on hits
                motion.add_impulse([0, 14, 15], np.array([0, -2 * onset_intensity, 0]))
            
            pose_sequence.append(joints.copy())
        
        result = {
            "poses": pose_sequence,
            "frame_count": frame_count,
            "joint_names": JOINT_NAMES,
            "limb_seq": LIMB_SEQ,
            "bone_colors": BONE_COLORS,
        }
        
        return (result,)


# ============================================================================
# SCAIL POSE RENDERER
# ============================================================================

class SCAILPoseRenderer:
    """Render pose sequence to SCAIL-style cylinder images.
    
    IMPORTANT: SCAIL expects pose renders at HALF the target video resolution.
    If generating 512x768 video, render poses at 256x384.
    Enable 'auto_half_resolution' to automatically halve the input dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_sequence": ("SCAIL_POSE_SEQUENCE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "Target VIDEO width (will be halved for pose render if auto_half enabled)"}),
                "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "Target VIDEO height (will be halved for pose render if auto_half enabled)"}),
                "auto_half_resolution": ("BOOLEAN", {"default": True,
                    "tooltip": "Automatically render at half resolution (required for SCAIL)"}),
                "fov": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Field of view in degrees"}),
                "cylinder_pixel_radius": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Cylinder thickness in approximate pixels (will be converted to world units)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render"
    CATEGORY = "SCAIL-AudioReactive"
    
    def render(self, pose_sequence, width, height, auto_half_resolution, fov, cylinder_pixel_radius):
        import taichi as ti
        
        # Apply half resolution for SCAIL compatibility
        if auto_half_resolution:
            render_width = width // 2
            render_height = height // 2
        else:
            render_width = width
            render_height = height
        
        # Initialize taichi if needed
        try:
            ti.init(arch=ti.gpu, default_fp=ti.f32)
        except:
            try:
                ti.init(arch=ti.cpu, default_fp=ti.f32)
            except:
                pass  # Already initialized
        
        poses = pose_sequence["poses"]
        limb_seq = pose_sequence["limb_seq"]
        bone_colors = pose_sequence["bone_colors"]
        
        # Calculate camera intrinsics from FOV (based on render resolution)
        fov_radians = fov * np.pi / 180
        larger_side = max(render_height, render_width)
        focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
        
        # Estimate average depth from poses to convert pixel radius to world radius
        all_depths = []
        for joints in poses:
            for j in range(len(joints)):
                if isinstance(joints[j], np.ndarray):
                    z = joints[j][2]
                else:
                    z = joints[j][2] if len(joints[j]) > 2 else 800
                if z > 0:
                    all_depths.append(z)
        
        avg_depth = np.mean(all_depths) if all_depths else 800.0
        
        # Convert pixel radius to world radius: world_radius = pixel_radius * depth / focal_length
        cylinder_radius = cylinder_pixel_radius * avg_depth / focal_length
        
        print(f"[SCAIL-AudioReactive] Render resolution: {render_width}x{render_height}")
        print(f"[SCAIL-AudioReactive] Focal length: {focal_length:.1f}, Avg depth: {avg_depth:.1f}")
        print(f"[SCAIL-AudioReactive] Pixel radius: {cylinder_pixel_radius} -> World radius: {cylinder_radius:.2f}")
        
        # If rendering at half resolution, we need to scale the pose coordinates
        # The poses were computed assuming full resolution camera center
        # Scale factor for X,Y coordinates (Z stays the same)
        if auto_half_resolution:
            scale_factor = 0.5
        else:
            scale_factor = 1.0
        
        # Build cylinder specs for each frame
        specs_list = []
        for joints in poses:
            frame_specs = []
            for limb_idx, (start_joint, end_joint) in enumerate(limb_seq):
                start_pos = joints[start_joint].copy() if isinstance(joints[start_joint], np.ndarray) else np.array(joints[start_joint])
                end_pos = joints[end_joint].copy() if isinstance(joints[end_joint], np.ndarray) else np.array(joints[end_joint])
                
                # Skip if joint is at origin (invalid)
                if np.sum(np.abs(start_pos)) < 1e-6 or np.sum(np.abs(end_pos)) < 1e-6:
                    continue
                
                # Scale X and Y for half resolution rendering
                start_pos[0] *= scale_factor
                start_pos[1] *= scale_factor
                end_pos[0] *= scale_factor
                end_pos[1] *= scale_factor
                
                color = bone_colors[limb_idx]
                frame_specs.append((
                    start_pos.tolist(),
                    end_pos.tolist(),
                    color
                ))
            specs_list.append(frame_specs)
        
        # Use the taichi renderer at half resolution
        frames_rgba = render_whole(
            specs_list,
            H=render_height,
            W=render_width,
            fx=focal_length,
            fy=focal_length,
            cx=render_width / 2,
            cy=render_height / 2,
            radius=cylinder_radius
        )
        
        # Convert to torch tensor (B, H, W, C) normalized to 0-1
        frames = []
        for frame in frames_rgba:
            frame_float = frame.astype(np.float32) / 255.0
            frames.append(frame_float[:, :, :3])  # Drop alpha
        
        images = np.stack(frames, axis=0)
        images = torch.from_numpy(images).float()
        
        return (images,)


# ============================================================================
# POSE SEQUENCE UTILITIES
# ============================================================================

class SCAILPoseFromNLF:
    """Convert NLF pose output to SCAIL pose format."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_poses": ("NLF_POSES",),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("pose_sequence",)
    FUNCTION = "convert"
    CATEGORY = "SCAIL-AudioReactive"
    
    def convert(self, nlf_poses):
        """Convert NLF output (24 joints) to COCO format (18 joints)."""
        
        # NLF to COCO mapping
        mapping = {
            15: 0,   # head -> nose
            12: 1,   # neck
            17: 2,   # left shoulder -> r_shoulder (mirrored)
            16: 5,   # right shoulder -> l_shoulder
            19: 3,   # left elbow -> r_elbow
            18: 6,   # right elbow -> l_elbow
            21: 4,   # left hand -> r_wrist
            20: 7,   # right hand -> l_wrist
            2: 8,    # left pelvis -> r_hip
            1: 11,   # right pelvis -> l_hip
            5: 9,    # left knee -> r_knee
            4: 12,   # right knee -> l_knee
            8: 10,   # left feet -> r_ankle
            7: 13,   # right feet -> l_ankle
        }
        
        poses = []
        for frame_data in nlf_poses:
            if isinstance(frame_data, torch.Tensor):
                joints_24 = frame_data.cpu().numpy()
            else:
                joints_24 = np.array(frame_data)
            
            joints_18 = np.zeros((18, 3), dtype=np.float32)
            for src, dst in mapping.items():
                if src < len(joints_24):
                    joints_18[dst] = joints_24[src]
            
            poses.append(joints_18)
        
        result = {
            "poses": poses,
            "frame_count": len(poses),
            "joint_names": JOINT_NAMES,
            "limb_seq": LIMB_SEQ,
            "bone_colors": BONE_COLORS,
        }
        
        return (result,)


# ============================================================================
# REFERENCE IMAGE POSE ALIGNMENT
# ============================================================================

class SCAILPoseFromDWPose:
    """
    Extract base pose from DWPose/VitPose detection on reference image.
    This ensures the generated pose sequence starts aligned with the reference.
    Works with output from Kijai's SCAIL-Pose 'Pose Detection VitPose to DWPose' node,
    or standard POSE_KEYPOINT format from other pose detection nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "image_height": ("INT", {"default": 768, "min": 64, "max": 4096}),
                "depth": ("FLOAT", {"default": 800.0, "min": 200.0, "max": 2000.0, 
                    "tooltip": "Estimated depth (Z distance) for 3D projection"}),
                "fov": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 120.0,
                    "tooltip": "Field of view matching the renderer"}),
            },
            "optional": {
                "dw_poses": ("DWPOSES",),  # From SCAIL-Pose VitPose to DWPose node
                "pose_keypoint": ("POSE_KEYPOINT",),  # Standard DWPose format
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE",)
    RETURN_NAMES = ("base_pose",)
    FUNCTION = "extract"
    CATEGORY = "SCAIL-AudioReactive"
    
    def extract(self, image_width, image_height, depth, fov, dw_poses=None, pose_keypoint=None):
        """
        Convert 2D DWPose keypoints to 3D pose by back-projecting with estimated depth.
        """
        
        # Use whichever input is provided
        dw_pose = dw_poses if dw_poses is not None else pose_keypoint
        
        if dw_pose is None:
            return self._default_pose(depth)
        
        # DWPose format: list of dicts with 'bodies', 'hands', 'faces'
        # bodies['candidate'] is list of (18, 3) arrays [x_norm, y_norm, confidence]
        
        if isinstance(dw_pose, list) and len(dw_pose) > 0:
            pose_data = dw_pose[0]  # First frame
        else:
            pose_data = dw_pose
        
        # Handle different possible structures
        keypoints_2d = None
        
        if isinstance(pose_data, dict):
            if 'bodies' in pose_data and 'candidate' in pose_data['bodies']:
                candidates = pose_data['bodies']['candidate']
                if isinstance(candidates, list) and len(candidates) > 0:
                    keypoints_2d = np.array(candidates[0])  # First person
                elif isinstance(candidates, np.ndarray) and len(candidates) > 0:
                    keypoints_2d = candidates[0] if candidates.ndim > 2 else candidates
            elif 'candidate' in pose_data:
                candidates = pose_data['candidate']
                if isinstance(candidates, list) and len(candidates) > 0:
                    keypoints_2d = np.array(candidates[0])
                elif isinstance(candidates, np.ndarray):
                    keypoints_2d = candidates[0] if candidates.ndim > 2 else candidates
        elif isinstance(pose_data, np.ndarray):
            keypoints_2d = pose_data
        elif isinstance(pose_data, torch.Tensor):
            keypoints_2d = pose_data.cpu().numpy()
            
        if keypoints_2d is None:
            print("[SCAIL-AudioReactive] Could not extract keypoints from pose data")
            return self._default_pose(depth)
        
        # Debug output
        print(f"[SCAIL-AudioReactive] keypoints_2d shape: {keypoints_2d.shape}")
        print(f"[SCAIL-AudioReactive] keypoints_2d sample (first 3): {keypoints_2d[:3]}")
        
        # Determine if coordinates are normalized (0-1) or pixel space
        # Check the range of x and y values
        x_vals = keypoints_2d[:, 0]
        y_vals = keypoints_2d[:, 1]
        x_max = x_vals[x_vals > 0].max() if np.any(x_vals > 0) else 0
        y_max = y_vals[y_vals > 0].max() if np.any(y_vals > 0) else 0
        
        # If max value > 1.5, assume pixel coordinates already
        is_normalized = x_max <= 1.5 and y_max <= 1.5
        
        print(f"[SCAIL-AudioReactive] x_max: {x_max:.3f}, y_max: {y_max:.3f}")
        print(f"[SCAIL-AudioReactive] Coordinates are: {'normalized (0-1)' if is_normalized else 'pixel space'}")
        
        # Calculate camera intrinsics
        fov_radians = fov * np.pi / 180
        larger_side = max(image_height, image_width)
        focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
        
        cx = image_width / 2
        cy = image_height / 2
        
        # Back-project 2D to 3D
        joints_3d = np.zeros((18, 3), dtype=np.float32)
        
        for i in range(min(18, len(keypoints_2d))):
            kp = keypoints_2d[i]
            
            # Handle both (x, y) and (x, y, conf) formats
            if len(kp) >= 3:
                x_val, y_val, conf = kp[0], kp[1], kp[2]
            else:
                x_val, y_val = kp[0], kp[1]
                conf = 1.0  # Assume valid if no confidence provided
            
            # Skip invalid keypoints
            if x_val <= 0 and y_val <= 0:
                continue
            if conf < 0.1:
                continue
            
            # Convert to pixel coords if normalized
            if is_normalized:
                x_px = x_val * image_width
                y_px = y_val * image_height
            else:
                x_px = x_val
                y_px = y_val
            
            # Back-project: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
            X = (x_px - cx) * depth / focal_length
            Y = (y_px - cy) * depth / focal_length
            Z = depth
            
            joints_3d[i] = [X, Y, Z]
        
        print(f"[SCAIL-AudioReactive] joints_3d neck: {joints_3d[1]}")
        
        # Fill in any missing joints with interpolation from neighbors
        joints_3d = self._interpolate_missing(joints_3d)
        
        pose = {
            "joints": torch.from_numpy(joints_3d).float(),
            "joint_names": JOINT_NAMES,
            "limb_seq": LIMB_SEQ,
        }
        
        return (pose,)
    
    def _interpolate_missing(self, joints):
        """Fill missing joints (zeros) by interpolating from parent/child joints."""
        
        # Check which joints are missing
        missing = np.all(joints == 0, axis=1)
        
        for i in range(18):
            if missing[i] and JOINT_PARENTS.get(i, -1) >= 0:
                parent = JOINT_PARENTS[i]
                if not missing[parent]:
                    # Use parent position with small offset
                    joints[i] = joints[parent] + np.array([0, 10, 0])
        
        return joints
    
    def _default_pose(self, depth):
        """Return a default relaxed pose if detection fails."""
        generator = SCAILBasePoseGenerator()
        return generator.generate("relaxed", 400.0, depth, 0.0, 0.0)


class SCAILPoseFromNLFSingle:
    """
    Extract a single base pose from NLF 3D pose detection on reference image.
    Use this when you have NLF output for your reference image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nlf_pose": ("NLF_POSE",),  # Single frame NLF output (24, 3)
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE",)
    RETURN_NAMES = ("base_pose",)
    FUNCTION = "extract"
    CATEGORY = "SCAIL-AudioReactive"
    
    def extract(self, nlf_pose):
        """Convert single NLF pose (24 joints) to SCAIL base pose (18 joints)."""
        
        # NLF to COCO mapping
        mapping = {
            15: 0,   # head -> nose
            12: 1,   # neck
            17: 2,   # left shoulder -> r_shoulder
            16: 5,   # right shoulder -> l_shoulder
            19: 3,   # left elbow -> r_elbow
            18: 6,   # right elbow -> l_elbow
            21: 4,   # left hand -> r_wrist
            20: 7,   # right hand -> l_wrist
            2: 8,    # left pelvis -> r_hip
            1: 11,   # right pelvis -> l_hip
            5: 9,    # left knee -> r_knee
            4: 12,   # right knee -> l_knee
            8: 10,   # left feet -> r_ankle
            7: 13,   # right feet -> l_ankle
        }
        
        if isinstance(nlf_pose, torch.Tensor):
            joints_24 = nlf_pose.cpu().numpy()
        else:
            joints_24 = np.array(nlf_pose)
        
        # Handle case where it's wrapped in list
        if joints_24.ndim == 3:
            joints_24 = joints_24[0]
        if joints_24.ndim == 3:
            joints_24 = joints_24[0]
            
        joints_18 = np.zeros((18, 3), dtype=np.float32)
        for src, dst in mapping.items():
            if src < len(joints_24):
                joints_18[dst] = joints_24[src]
        
        pose = {
            "joints": torch.from_numpy(joints_18).float(),
            "joint_names": JOINT_NAMES,
            "limb_seq": LIMB_SEQ,
        }
        
        return (pose,)


class SCAILAlignPoseToReference:
    """
    Align a generated/modified pose sequence to match the reference image pose.
    Applies offset so first frame matches the reference position.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_sequence": ("SCAIL_POSE_SEQUENCE",),
                "reference_pose": ("SCAIL_POSE",),
                "align_position": ("BOOLEAN", {"default": True, 
                    "tooltip": "Align center position to reference"}),
                "align_scale": ("BOOLEAN", {"default": False,
                    "tooltip": "Scale to match reference proportions"}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("aligned_sequence",)
    FUNCTION = "align"
    CATEGORY = "SCAIL-AudioReactive"
    
    def align(self, pose_sequence, reference_pose, align_position, align_scale):
        poses = pose_sequence["poses"]
        ref_joints = reference_pose["joints"].numpy()
        
        if len(poses) == 0:
            return (pose_sequence,)
        
        first_pose = poses[0]
        
        # Calculate centers (using neck as reference point)
        ref_center = ref_joints[1]  # Neck
        first_center = first_pose[1]  # Neck
        
        aligned_poses = []
        
        for pose in poses:
            new_pose = pose.copy()
            
            if align_position:
                # Offset to match reference position
                offset = ref_center - first_center
                new_pose = new_pose + offset
            
            if align_scale:
                # Calculate scale based on torso length (neck to hip midpoint)
                ref_hip_mid = (ref_joints[8] + ref_joints[11]) / 2
                ref_torso = np.linalg.norm(ref_joints[1] - ref_hip_mid)
                
                first_hip_mid = (first_pose[8] + first_pose[11]) / 2
                first_torso = np.linalg.norm(first_pose[1] - first_hip_mid)
                
                if first_torso > 1e-6:
                    scale = ref_torso / first_torso
                    # Scale around the neck
                    neck = new_pose[1].copy()
                    new_pose = (new_pose - neck) * scale + neck
            
            aligned_poses.append(new_pose)
        
        result = {
            "poses": aligned_poses,
            "frame_count": len(aligned_poses),
            "joint_names": pose_sequence["joint_names"],
            "limb_seq": pose_sequence["limb_seq"],
            "bone_colors": pose_sequence["bone_colors"],
        }
        
        return (result,)


class SCAILPosePreview:
    """Preview audio features as a simple visualization."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_features": ("SCAIL_AUDIO_FEATURES",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1024}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "SCAIL-AudioReactive"
    
    def preview(self, audio_features, width, height):
        frame_count = audio_features["frame_count"]
        
        rms = audio_features["rms"].numpy()
        bass = audio_features["bass"].numpy()
        mid = audio_features["mid"].numpy()
        treble = audio_features["treble"].numpy()
        
        # Create visualization image
        img = np.zeros((height, width, 3), dtype=np.float32)
        
        bar_height = height // 4
        
        for i in range(frame_count):
            x = int(i * width / frame_count)
            if x >= width:
                break
            
            # RMS - white
            h = int(rms[i] * bar_height)
            img[0:h, x] = [1.0, 1.0, 1.0]
            
            # Bass - red
            h = int(bass[i] * bar_height)
            img[bar_height:bar_height+h, x] = [1.0, 0.2, 0.2]
            
            # Mid - green
            h = int(mid[i] * bar_height)
            img[bar_height*2:bar_height*2+h, x] = [0.2, 1.0, 0.2]
            
            # Treble - blue
            h = int(treble[i] * bar_height)
            img[bar_height*3:bar_height*3+h, x] = [0.2, 0.2, 1.0]
        
        # Flip vertically so bars go up
        img = np.flip(img, axis=0).copy()
        
        return (torch.from_numpy(img).unsqueeze(0),)


class SCAILBeatPreview:
    """Preview beat detection as a visualization with beat markers."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beat_info": ("SCAIL_BEAT_INFO",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1024}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "SCAIL-AudioReactive"
    
    def preview(self, beat_info, width, height):
        frame_count = beat_info["frame_count"]
        beat_frames = beat_info["beat_frames"]
        downbeat_frames = beat_info["downbeat_frames"]
        onset_env = beat_info["onset_envelope"].numpy()
        energy = beat_info["energy"].numpy()
        tempo = beat_info["tempo"]
        
        # Create visualization image
        img = np.zeros((height, width, 3), dtype=np.float32)
        
        # Background grid - subtle
        for i in range(0, width, 50):
            img[:, i:i+1] = [0.1, 0.1, 0.1]
        
        # Draw energy curve (blue area)
        for i in range(frame_count):
            x = int(i * width / frame_count)
            if x >= width:
                break
            
            h = int(energy[i] * height * 0.4)
            img[height-h:height, x] = [0.2, 0.2, 0.6]
        
        # Draw onset envelope (green line)
        for i in range(frame_count):
            x = int(i * width / frame_count)
            if x >= width:
                break
            
            h = int(onset_env[i] * height * 0.6)
            y = height - h - int(height * 0.3)
            if 0 <= y < height:
                img[max(0,y-1):min(height,y+2), x] = [0.2, 0.8, 0.2]
        
        # Draw beat markers (yellow vertical lines)
        for beat_frame in beat_frames:
            x = int(beat_frame * width / frame_count)
            if 0 <= x < width:
                img[:, max(0,x-1):min(width,x+2)] = [0.9, 0.9, 0.2]
        
        # Draw downbeat markers (red, thicker)
        for db_frame in downbeat_frames:
            x = int(db_frame * width / frame_count)
            if 0 <= x < width:
                img[:, max(0,x-2):min(width,x+3)] = [1.0, 0.3, 0.3]
        
        # Add tempo text area (top-left corner)
        # Simple block to indicate tempo range
        tempo_bar_width = int(min(tempo / 200.0, 1.0) * 100)
        img[5:15, 5:5+tempo_bar_width] = [0.8, 0.4, 0.8]
        
        return (torch.from_numpy(img).unsqueeze(0),)


# ============================================================================
# NODE MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SCAILAudioFeatureExtractor": SCAILAudioFeatureExtractor,
    "SCAILBasePoseGenerator": SCAILBasePoseGenerator,
    "SCAILBeatDetector": SCAILBeatDetector,
    "SCAILBeatDrivenPose": SCAILBeatDrivenPose,
    "SCAILPoseRenderer": SCAILPoseRenderer,
    "SCAILPoseFromNLF": SCAILPoseFromNLF,
    "SCAILPoseFromNLFSingle": SCAILPoseFromNLFSingle,
    "SCAILPoseFromDWPose": SCAILPoseFromDWPose,
    "SCAILAlignPoseToReference": SCAILAlignPoseToReference,
    "SCAILPosePreview": SCAILPosePreview,
    "SCAILBeatPreview": SCAILBeatPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILAudioFeatureExtractor": "SCAIL Audio Feature Extractor",
    "SCAILBasePoseGenerator": "SCAIL Base Pose Generator",
    "SCAILBeatDetector": "SCAIL Beat Detector",
    "SCAILBeatDrivenPose": "SCAIL Beat-Driven Dance Pose",
    "SCAILPoseRenderer": "SCAIL Pose Renderer",
    "SCAILPoseFromNLF": "SCAIL Pose Sequence from NLF",
    "SCAILPoseFromNLFSingle": "SCAIL Base Pose from NLF",
    "SCAILPoseFromDWPose": "SCAIL Base Pose from DWPose",
    "SCAILAlignPoseToReference": "SCAIL Align Pose to Reference",
    "SCAILPosePreview": "SCAIL Audio Features Preview",
    "SCAILBeatPreview": "SCAIL Beat Detection Preview",
}
