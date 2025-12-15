# ComfyUI-SCAIL-AudioReactive

Generate audio-reactive SCAIL pose sequences for character animation without requiring input video tracking.

## What it does

Creates SCAIL-compatible 3D skeleton pose renders driven by audio analysis. Instead of extracting poses from a driving video, this generates poses procedurally using:

1. **Beat detection** — Poses hit on musical beats
2. **Keyframe interpolation** — Smooth transitions between dance poses
3. **Audio feature modulation** — Bass, mid, treble, onsets drive continuous motion overlay
4. **Joint rotation system** — Proper hierarchical skeleton articulation

## Nodes

### Audio Analysis
- **SCAILAudioFeatureExtractor** — Extracts per-frame RMS, bass, mid, treble, and onset features
- **SCAILBeatDetector** — Detects beats, downbeats, tempo from audio using librosa

### Base Pose Generation
- **SCAILBasePoseGenerator** — Creates T-pose, A-pose, or relaxed starting poses
- **SCAILPoseFromDWPose** — Extract base pose from DWPose detection (2D→3D backprojection)
- **SCAILPoseFromNLFSingle** — Extract base pose from NLF 3D detection

### Animation
- **SCAILBeatDrivenPose** — Main dance generator: beat-aligned keyframes + audio modulation
- **SCAILAlignPoseToReference** — Align generated pose sequence to match reference position/scale

### Rendering
- **SCAILPoseRenderer** — Render pose sequence to SCAIL-style 3D cylinder images
- **SCAILPosePreview** — Visualize extracted audio features
- **SCAILBeatPreview** — Visualize detected beats and energy

### Conversion
- **SCAILPoseFromNLF** — Convert NLF pose sequence to SCAIL format

## Workflow

```
                                    ┌─── SCAILAudioFeatureExtractor ◄─┐
                                    │                                 │
Audio ──┬── SCAILBeatDetector ──────┼─────────────────────────────────┤
        │                           │                                 │
        └───────────────────────────┘                                 │
                                                                      │
RefImage ──► DWPose ──► SCAILPoseFromDWPose ──┐                       │
                                              │                       │
                                              ▼                       │
                                    SCAILBeatDrivenPose ◄─────────────┘
                                              │
                                              ▼
                                    SCAILAlignPoseToReference
                                              │
                                              ▼
                                    SCAILPoseRenderer
                                              │
                                              ▼
                                    IMAGE → WanSCAILPoseEmbeds
```

## Installation

1. Clone to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-SCAIL-AudioReactive
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install taichi librosa
```

## Parameters

### SCAILBeatDetector
- `frame_count` — Number of frames to generate (match your video length)
- `fps` — Frames per second

### SCAILAudioFeatureExtractor
- `frame_count` — Number of frames (should match beat detector)
- `fps` — Frames per second
- `bass_range` — Frequency range for bass (default: 20-250 Hz)
- `mid_range` — Frequency range for mids (default: 250-2000 Hz)
- `treble_range` — Frequency range for treble (default: 2000-8000 Hz)
- `smoothing` — Temporal smoothing factor

### SCAILBeatDrivenPose

**Keyframe Control:**
- `energy_style` — auto/low/medium/high — affects pose selection
- `pose_intensity` — Scale of keyframe pose rotations
- `anticipation_frames` — Start moving before the beat lands
- `hold_frames` — Hold pose at peak before transitioning
- `pose_variation` — Randomness in pose selection
- `seed` — Random seed for reproducibility

**Audio Modulation Intensities:**
- `bass_intensity` — Bass → bounce, knee bend (alternating legs)
- `mid_intensity` — Mid → horizontal sway, hip motion
- `treble_intensity` — Treble → arm raise, elbow variation
- `onset_intensity` — Onsets → head snap on transients

### SCAILPoseRenderer
- `width`, `height` — Output dimensions (use HALF your target video resolution)
- `cylinder_pixel_radius` — Thickness of skeleton bones

## Dance Poses

Built-in pose library includes:
- `neutral` — Relaxed standing
- `arms_up` — Both arms raised
- `arms_out` — Arms spread wide
- `crouch` — Knees bent, lowered stance
- `pump_right` / `pump_left` — Fist pump
- `lean_left` / `lean_right` — Body lean
- `hip_left` / `hip_right` — Hip shift
- `step_left` / `step_right` — Weight shift (feet stay grounded)
- `head_down` / `head_back` — Head tilt
- `look_left` / `look_right` — Head turn
- `arms_crossed_low` — Arms crossed in front

Poses are combined into sequences based on energy level (low/medium/high).

## Audio Feature → Body Part Mapping

| Feature | Drives |
|---------|--------|
| **Bass** | Vertical bounce, knee bend (alternating), forward lean |
| **Mid** | Horizontal sway, hip tilt, spine tilt |
| **Treble** | Arm raise, elbow bend variation |
| **Onset** | Head snap on transients |

## Technical Details

### Joint Rotation System
Poses use proper hierarchical rotation — rotating a shoulder moves the entire arm chain (elbow, wrist). This produces more natural articulation than simple position offsets.

### Ground Constraint
The system keeps feet planted by:
1. Alternating leg phases (one leg always more bent)
2. Applying bounce only to upper body
3. Correcting any ankle lift above threshold

### COCO 18-Joint Format
Output uses standard COCO skeleton:
```
0:nose  1:neck  2:r_shoulder  3:r_elbow  4:r_wrist
5:l_shoulder  6:l_elbow  7:l_wrist  8:r_hip  9:r_knee
10:r_ankle  11:l_hip  12:l_knee  13:l_ankle  14:r_eye
15:l_eye  16:r_ear  17:l_ear
```

## Usage with SCAIL

The output IMAGE from SCAILPoseRenderer feeds directly into Kijai's WanVideoWrapper SCAIL nodes:
- `WanVideoAddSCAILPoseEmbeds` — Takes the rendered pose images
- `WanVideoAddSCAILReferenceEmbeds` — Takes your reference character image

**Important:** SCAIL expects pose renders at HALF your target video resolution.

## Credits

- SCAIL: [zai-org/SCAIL](https://github.com/zai-org/SCAIL)
- Taichi renderer from [zai-org/SCAIL-Pose](https://github.com/zai-org/SCAIL-Pose)
- ComfyUI integration: [kijai/ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose)
- Beat detection: [librosa](https://librosa.org/)
