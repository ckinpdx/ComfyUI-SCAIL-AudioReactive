# ComfyUI-SCAIL-AudioReactive

Generate audio-reactive SCAIL pose sequences for character animation without requiring input video tracking. Now supports **Multi-Character Choreography** and **AIST++ Motion Capture Dance Data**.

## What it does

Creates SCAIL-compatible 3D skeleton pose renders driven by audio analysis. Instead of extracting poses from a driving video, this generates poses procedurally using:

1. **Beat detection** — Poses hit on musical beats
2. **Multi-Character Sync** — Orchestrate groups with Unison, Mirror, or Random interactions
3. **Physics Simulation** — Momentum, drag, and "Sticky Feet" constraints for grounded motion
4. **Auto-Rigging** — Automatically corrects "non-neutral" reference poses (e.g., character starting with foot up)
5. **Dynamic Scaling** — Adjusts movement amplitude based on character size/distance to prevent collisions
6. **AIST++ Dance Library** — Real motion capture dance data from professional dancers across 10 genres

## Nodes

### Audio Analysis
- **SCAILAudioFeatureExtractor** — Extracts per-frame RMS, bass, mid, treble, and onset features.
- **SCAILBeatDetector** — Detects beats, downbeats, and tempo from audio using librosa.

### Base Pose Generation
- **SCAILBasePoseGenerator** — Procedurally create 1 to 5 skeletons (T-pose, Active Idle) with adjustable spacing.
- **SCAILPoseFromDWPose** — Extract base poses from reference images. Supports **1:1 fidelity** for multiple characters (detects everyone in the image, no cloning).

### Animation
- **SCAILBeatDrivenPose** — The Choreographer. Generates physics-based dance sequences.
  - Supports **Interaction Modes** (Mirror, Unison, Random).
  - Handles **Anti-Jelly Bone** constraints to keep limbs rigid.
  - Includes **80+ Dance Poses** (Hip Hop, Rock, Pop, etc.)
- **SCAILAlignPoseToReference** — Align generated pose sequence to match reference position/scale.

### AIST++ Dance Library (NEW)
Real motion capture data from the AIST++ dataset — professional dancers performing choreographed routines across 10 genres. Auto-downloads from HuggingFace on first use.

**Chunk-Based (Reactive):**
- **SCAILAISTLibraryLoader** — Loads the chunked dance library (~220MB, auto-downloads).
- **SCAILAISTBeatDance** — Beat-triggered chunk selection based on audio energy.
  - Mix multiple genres (break, pop, lock, etc.)
  - Energy-aware: low/mid/high energy chunks matched to audio
  - Chunks chain for longer sequences
  - **Note:** Can be janky at transitions since chunks are cut at velocity minimums, not perfect phrase boundaries. Good for variety and reactivity, less smooth than full sequences.
- **SCAILAISTChunkPreview** — Preview individual chunks by genre/energy.

**Full Sequence (Smooth):**
- **SCAILAISTFullLoader** — Loads full dance sequences (~200MB, auto-downloads).
- **SCAILAISTFullSequence** — Plays complete choreographed dances.
  - Single genre dropdown (no mixing)
  - Smooth, continuous motion from real performances
  - **Note:** Can feel repetitive if you recognize the source choreography. Best for shorter generations or when you want authentic, uncut dance motion.

**Available Genres:** break, pop, lock, waack, krump, house, street_jazz, ballet_jazz, la_hip_hop, middle_hip_hop

### Rendering
- **SCAILPoseRenderer** — Render pose sequence to SCAIL-style 3D cylinder images.
- **SCAILPosePreview** — Visualize extracted audio features.
- **SCAILBeatPreview** — Visualize detected beats and energy.

## Installation

1. Clone to your ComfyUI custom_nodes folder:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/ckinpdx/ComfyUI-SCAIL-AudioReactive
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

AIST++ data downloads automatically on first use (~200-220MB per library).

## Parameters

### SCAILBasePoseGenerator
- `character_count` — Generate 1 to 5 procedural skeletons side-by-side.
- `spacing` — Distance between characters in world units.

### SCAILBeatDrivenPose

**Interaction & Style:**
- `dance_style`:
    - **auto**: Selects moves based on audio energy levels.
    - **hip_hop, rock, disco, etc**: Forces specific genre move sets.
- `interaction_mode`:
    - **unison**: All characters perform the exact same move at the same time.
    - **mirror**: Characters swap Left/Right moves based on the leader (Center/Left).
    - **random**: Every character picks a distinct move from the style bucket.
- `energy_style` — auto/low/medium/high.
- `motion_smoothness` — Higher = floatier/smoother. Lower = snappier/robotic.
- `anticipation` — Number of frames to start moving *before* the beat hits.

**Audio Modulation:**
- `groove_amount` — Intensity of the continuous hip sway/figure-8 loop.
- `bass_intensity` — Bass → vertical bounce (scaled to body size).
- `treble_intensity` — Treble → arm/hand jitter.

### SCAILAISTBeatDance (Chunk Mode)
- `sync_dancers` — All dancers do same moves (true) or independent (false).
- `genre_*` toggles — Enable/disable each of the 10 dance genres.
- `transition_frames` — Frames to blend between chunks (smooths transitions).
- `chunks_per_beat` — Chain consecutive chunks (1=0.5s, 2=1s, etc.).
- `energy_sensitivity` — How strongly audio energy affects chunk selection.

### SCAILAISTFullSequence (Full Sequence Mode)
- `genre` — Single genre dropdown.
- `transition_frames` — Frames to blend from reference pose to dance start.
- `start_beat` — Which beat in the source to start from (0=beginning).

### SCAILPoseFromDWPose
- **Note:** This node strictly extracts what is detected in your reference image. It does not clone characters. To animate 3 people, you need a reference image with 3 people.

## Multi-Character Tips

If you are using a reference image with multiple people (e.g., a band or dance troupe), you must ensure your **Upstream DWPose/OpenPose Detector** actually finds them all.

1. **Resolution:** Set upstream detector resolution to **1024** or higher for group shots.
2. **Model:** Use **`yolo_nas_l_fp16.onnx`** or **`yolox_x.onnx`** for the BBox detector. Standard `yolox_l` often misses people in complex poses (e.g., arms touching) or back rows.
3. **Max People:** Ensure the upstream node allows `max_people > 1` (if using OpenPose).

## AIST++ vs Procedural Animation

| Feature | Procedural (BeatDrivenPose) | AIST++ Chunks | AIST++ Full |
|---------|----------------------------|---------------|-------------|
| Motion Quality | Synthetic, can look robotic | Real mocap, natural | Real mocap, natural |
| Variety | 80+ poses, infinite combos | 10 genres, ~44k chunks | 10 genres, ~1400 sequences |
| Audio Reactivity | High (every beat) | Medium (energy-based) | Low (just plays) |
| Transitions | Smooth (physics-based) | Can be janky | Smooth (continuous) |
| Floor Work | Limited | Full (breakdance etc) | Full |
| Best For | Stylized, rhythmic loops | Reactive, varied | Authentic, cinematic |

## Technical Details

### "Sticky Feet" Physics
The custom `MotionDynamics` engine uses variable drag coefficients. Ankles have **0 drag** and **2x stiffness**, forcing them to snap to the floor position unless a specific dance move lifts them. This prevents the "floating" effect common in procedural animation.

### Neutral Structure Initialization (Auto-Rigging)
If your reference image has a character standing on one leg or mid-stride, the system calculates a mathematical "Neutral Standing" skeleton based on their bone lengths. The physics engine then interpolates from the Reference Pose -> Neutral Pose -> Dance Move, preventing the character from getting stuck in their initial pose (e.g., keeping one foot in the air).

### Scale-Aware Movement
Movements are normalized based on the character's torso length. A small character in the background will perform smaller absolute movements than a character in the foreground, maintaining correct perspective and preventing characters from colliding.

### AIST++ Data Processing
The AIST++ dance data is converted from COCO 17-keypoint format to OpenPose 18-keypoint format for SCAIL compatibility. Chunks are cut at velocity minimums (natural pauses in movement) rather than fixed intervals to preserve move integrity. Energy tagging uses velocity + acceleration metrics, calculated per-genre to account for style differences (krump is naturally higher energy than house).



## Credits

- SCAIL: [zai-org/SCAIL](https://github.com/zai-org/SCAIL)
- Taichi renderer from [zai-org/SCAIL-Pose](https://github.com/zai-org/SCAIL-Pose)
- ComfyUI integration: [kijai/ComfyUI-SCAIL-Pose](https://github.com/kijai/ComfyUI-SCAIL-Pose)
- Beat detection: [librosa](https://librosa.org/)
- Expanded Pose Library: Discord user **NebSH**


## Legal & Licensing

This software downloads and processes third-party datasets. Users are responsible for complying with the attribution requirements of these licenses when publishing generated content.
1. CMU Motion Capture Database

    Source: mocap.cs.cmu.edu

    License: Free for use (including commercial), but requires citation.

    Required Citation:

        "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217."

2. AIST++ Dataset

    Source: Google Research AIST++

    License: CC-BY 4.0 International

    Terms: Commercial use allowed. Modification allowed. Redistribution allowed.

    Required Attribution: You must give appropriate credit to the original researchers (Li et al., ICCV 2021) in your derived works (videos/images).

Disclaimer

The authors of SCAIL are not responsible for end-user license violations. This tool is provided "AS IS" without warranty of any kind. Access to the datasets is facilitated for research and creative toolchain integration; no ownership of the raw data is claimed.
