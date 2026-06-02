# WordleBot RS2

> A UR3e robotic system that autonomously plays Wordle using computer vision, reinforcement learning, and pick-and-place manipulation.

**41118 AI in Robotics · University of Technology Sydney · 2025**

| Team Member | Subsystem | Focus |
|---|---|---|
| Luke Klusman | SS1 — Perception | RealSense D435i, CNN letter detection, TF2 world-frame projection |
| Connor Lindsell | SS2 — Pick & Place | PPO path planning, UR3e motion execution, hl_control |
| Elijah Spannenberg | SS3 — GUI & Orchestration | Qt5 GUI, mission coordinator behaviour tree, voice control |
| James Farrell | SS4 — Game Logic | Wordle solver, gamification node, pre/post-motion confirmation |

📽 **Portfolio:** https://lukefklusman.github.io/AI_WordleBot_Submission/

---

## How it works

1. The RealSense D435i scans a workspace of lettered cards using a brightness threshold detector.
2. A trained CNN (36 classes: A–Z, 0–9) classifies each letter across 15 frames of temporal smoothing.
3. A corner dot on each card disambiguates 180° orientation ambiguity and suppresses upside-down reads.
4. Block positions are projected from camera frame to robot world frame via the live UR3e TF tree.
5. The word solver selects an optimal guess from a 14,855-word dictionary using frequency-ranked scoring.
6. Pre-motion confirmation verifies all required letters are visible before dispatching to the robot.
7. The UR3e arm executes pick-and-place to physically spell the word on the board.
8. Wordle feedback (G/B/I) prunes the candidate list. Repeat until solved or 6 attempts reached.

---

## Repository structure

```
RS2/
├── perception/
│   ├── src/
│   │   ├── realsense_camera_cnn.py   # SS1: RealSense + CNN detection node
│   │   ├── collect_training_data.py  # Data collection (RealSense or webcam)
│   │   ├── train_letter_cnn.py       # CNN training script
│   │   └── webcam_node.py            # Webcam substitute (no RealSense needed)
│   ├── launch/
│   │   └── ss1_perception.launch.py  # RealSense + perception + gamification
│   ├── data/raw/                     # Training images (A-Z, 0-9 subdirectories)
│   └── outputs/
│       ├── letter_cnn.pt             # Trained model weights
│       ├── confusion_matrix.png      # Clean test confusion matrix
│       ├── confusion_matrix_hard.png # Hard augmented test confusion matrix
│       ├── training_curves.png       # Loss + accuracy over epochs
│       └── training_report.txt       # Per-class precision/recall report
├── gamification/
│   ├── gamification_node.py          # SS4: Wordle solver ROS2 node
│   ├── gamification_trial.py         # SS4: With pre/post-motion confirmation
│   ├── wordle_logic.py               # Core algorithm (scoring, filtering, guessing)
│   ├── dictionary.py                 # Word list loader
│   ├── dictionary.txt                # 14,855 valid 5-letter words
│   └── constants.py                  # Shared constants + easter egg words
├── interaction_execution/            # SS3: Qt5 GUI + mission coordinator
│   ├── src/
│   │   ├── main_window.cpp           # Main Qt5 GUI window
│   │   ├── mission_coordinator.cpp   # Behaviour-tree coordinator node
│   │   ├── camera_view.cpp           # Camera feed widget
│   │   ├── hl_digital_twin_view.cpp  # Pick/place task visualiser
│   │   ├── wordle_view.cpp           # Embedded Wordle board (QWebEngine)
│   │   └── wordle-clone/             # HTML/JS Wordle board
│   ├── include/
│   └── launch/
│       └── gui.launch.py             # Launch GUI + optional subsystems
├── voice_control/
│   ├── main.py                       # Voice-controlled game loop
│   └── speaker_verification.py       # MFCC voiceprint registration
└── motion_planning_and_control/      # SS2: RL + pick-and-place (Connor)
    └── hl_control/
        └── msg/
            ├── GameboardState.msg    # Custom msg: list of detected letter blocks
            └── LetterObject.msg      # Custom msg: single block + letter + pose
```

---

## Prerequisites

**System:**
- Ubuntu 22.04
- ROS 2 Humble Hawksbill
- Python 3.10+

**Hardware:**
- UR3e collaborative robot arm (IP: `192.168.0.191` on lab network)
- OnRobot RG2 gripper
- Intel RealSense D435i (USB 3.0)

**ROS 2 packages:**
```bash
sudo apt install ros-humble-realsense2-camera
sudo apt install ros-humble-ur-robot-driver
sudo apt install ros-humble-moveit
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-tf2-tools
sudo apt install ros-humble-rviz2
```

**Python packages:**
```bash
# Inference + detection
pip install torch torchvision
pip install opencv-python
pip install pyrealsense2

# Training only
pip install scikit-learn matplotlib seaborn

# Robot control
pip install stable-baselines3

# Voice control (optional)
pip install SpeechRecognition sounddevice scipy librosa soundfile
```

**Qt5 (SS3 GUI):**
```bash
sudo apt install qtbase5-dev qtwebengine5-dev
```

---

## Build

```bash
# Clone the repository
git clone https://github.com/LukeFKlusman/WordleBot.git ~/ros2_ws/src/RS2
cd ~/ros2_ws

# Build custom message types first (required by perception and GUI)
colcon build --packages-select hl_control
source install/setup.bash

# Build remaining packages
colcon build --packages-select wordlebot_control interaction_execution
source install/setup.bash

# Set domain ID in every terminal — isolates from other groups on the lab network
export ROS_DOMAIN_ID=10
```

> **Important:** `hl_control` must be built before the perception node starts. Without it,
> `/perception/gameboard_state` will not be published and Connor's node will not receive
> world-frame block positions. The perception node logs a warning and continues without it.

---

## Running the system

### SS1 — Perception + Gamification (single launch)

This launch file starts the RealSense driver, waits 5 s, starts the CNN perception node,
then waits another 5 s before starting the gamification node.

```bash
# Standard launch
ros2 launch perception/launch/ss1_perception.launch.py

# With MediaPipe human detection enabled
ros2 launch perception/launch/ss1_perception.launch.py human_detection:=true
```

Startup sequence:

| Time | What starts |
|---|---|
| t = 0 s | RealSense driver — wait for `RealSense Node Is Up!` |
| t = 5 s | `realsense_camera_cnn.py` — CNN detection node |
| t = 10 s | `gamification_trial.py` — word solver node |

Controls in the perception window: `SPACEBAR` = toggle scanning · `Q` = quit

---

### SS2 — RL & Pick-and-Place (Connor Lindsell)

#### Overview

SS2 is responsible for training a reinforcement learning policy to control the UR3e arm
and deploying it for physical pick-and-place execution. The subsystem:

- Trains a **PPO (Proximal Policy Optimisation)** agent in a simulated environment
- Transfers the learned policy to the physical UR3e
- Subscribes to `/perception/gameboard_state` for world-frame block positions
- Subscribes to `/hl_control/word_request` for the target word from SS4
- Plans and executes a pick-and-place trajectory for each letter in sequence
- Reports motion completion via `/wordle_bot/mission_complete`

#### RL training

> ⚙️ **Connor — fill in this section** with:
> - Simulation environment used (PyBullet / Gazebo / custom)
> - Observation space definition (joint angles, end-effector pose, target position, etc.)
> - Action space (joint velocities / end-effector delta / torques)
> - Reward function design (distance to target, grasp success, time penalty, etc.)
> - Key PPO hyperparameters (learning rate, n_steps, clip_range, total_timesteps)
> - How sim-to-real transfer was handled (domain randomisation, calibration, etc.)
> - Training duration and final reward / success rate achieved

```python
# Placeholder — replace with actual training command / script
python3 motion_planning_and_control/train_rl_policy.py
```

#### Running SS2

> ⚙️ **Connor — fill in this section** with:
> - UR3e robot driver launch command (robot IP, URCap activation on teach pendant)
> - MoveIt / wordlebot_control launch command
> - Any additional nodes required (gripper driver, controller spawner, etc.)
> - Expected terminal output confirming the driver is ready
> - How the trained policy weights are loaded at runtime

```bash
# Placeholder — replace with actual SS2 launch commands
ros2 launch ur_onrobot_control start_robot.launch.py \
  ur_type:=ur3e \
  onrobot_type:=rg2 \
  robot_ip:=192.168.0.191 \
  launch_rviz:=false

ros2 launch wordlebot_control wordle_bot.launch.py
```

Wait for `joint_state_broadcaster` active before continuing. The TF tree must be
publishing for perception world-frame coordinates to work.

---

### SS3 — GUI & Mission Coordinator (Elijah Spannenberg)

#### Overview

SS3 is the operator-facing control system and orchestration layer. It provides a Qt5 C++ GUI with real-time camera feeds, an embedded Wordle board, system diagnostics, and voice control; it also runs a behaviour-tree-style mission coordinator node that synchronises all four subsystems, enforces safety constraints, and manages state transitions.

**Key responsibilities:**
- Operator interface (Qt5 GUI with RViz, camera view, diagnostics, voice commands)
- Mission orchestration via ROS 2 behaviour tree (500 ms tick rate)
- Safety monitoring (human detection, emergency stop, state validation)
- Perception lifecycle management (scan triggers, timeouts, retries)
- Game state feedback and diagnostics

#### GUI Components

The SS3 GUI is a single Qt5 C++ application with **four main views**:

1. **RViz Viewport** — 3D simulation of the UR3e, gripper, gameboard, and detected blocks (live TF tree rendering)
2. **HL Digital Twin View** — High-level task visualiser showing pick-and-place sequence and step status
3. **Camera View** — Live annotated camera feed from perception with detection overlays
4. **Diagnostics Panel** — Real-time game state, candidate words, confirmation status, mission progress

**Additional controls:**
- Voice control interface (push-to-talk, speaker verification status)
- Mode selector (Mode A: robot guesses / Mode B: human guesses)
- Scan trigger buttons and mode lock controls
- Emergency stop (E-stop) button and safety indicator
- Embedded HTML/JS Wordle board (rendered via QWebEngineView)

#### Mission Coordinator FSM

The mission coordinator runs as a C++ node with a **500 ms tick** and a priority-based behaviour tree structure:

```
Each tick evaluates in order:
  1. Safety guard (highest priority)
     - Monitors /perception/human_detected
     - Publishes /wordle_bot/stop_mission on human presence
     - Blocks all motion and resets to SAFETY_STOPPED
  
  2. Failure detection & recovery
     - Checks for perception timeout (configurable, e.g., 20 s)
     - Checks for motion timeout (configurable, e.g., 20 s)
     - Automatically retries perception scan (max configurable retries)
  
  3. State machine (10 states)
     IDLE → SCANNING → READY_TO_MOVE → MOVING → SCAN_BLOCKED → STOPPED → IDLE
     Plus error states: PERCEPTION_FAILED, MOTION_FAILED, SAFETY_STOPPED
  
  4. Command execution
     - Publishes /mission/state = SCANNING or IDLE
     - Publishes /wordle_bot/start_mission (pulsed 5× when dispatching motion)
  
  5. Status publication
     - Latches /wordle_bot/mission_state (current FSM state)
     - Publishes /wordle_bot/mission_progress (7-step plan with per-step status)
```

**State descriptions:**

| State | Trigger | Action | Next |
|---|---|---|---|
| IDLE | User START | Publish `/mission/state = SCANNING` | SCANNING |
| SCANNING | Gameboard received & locked | Check detections ready; timeout if > max_s | READY_TO_MOVE or PERCEPTION_FAILED |
| READY_TO_MOVE | All required letters detected | Publish `/wordle_bot/start_mission` (5×) | MOVING |
| MOVING | Motion in progress | Wait for `/wordle_bot/mission_complete` or timeout | STOPPED or MOTION_FAILED |
| STOPPED | Motion complete, no errors | Brief pause for post-motion confirmation | IDLE |
| PERCEPTION_FAILED | Scan timeout or max retries exceeded | Publish error diagnostics; await user retry | IDLE |
| MOTION_FAILED | Motion timeout | Publish error state; block further dispatch | IDLE |
| SAFETY_STOPPED | Human detected | Emergency stop; latch state | IDLE |

#### Voice Control

Voice control integrates **speech recognition** (via `SpeechRecognition` library) with **speaker verification** (MFCC voiceprint matching).

**Workflow:**
1. User enrolls voiceprint by speaking a fixed phrase 5 times (`speaker_verification.py`)
   - Extracts MFCC features and stores reference embedding
2. At runtime, user presses push-to-talk (or voice-activation trigger)
3. System captures audio, extracts MFCC, compares to enrolled voiceprint
4. If match confidence > threshold, command is accepted; else rejected with audio feedback

**Supported commands:**
- "Start scan" → Publish `/mission/state = SCANNING`
- "Stop" / "Reset" → Publish `/mission/state = IDLE`
- "Mode A" → Set to robot-guesses mode
- "Mode B" → Set to human-guesses mode
- "Guess [word]" → Submit player guess (Mode B)

Voice processing happens off-line (background thread); GUI shows "listening" indicator and confirmation of recognised command.

#### Running SS3

```bash
# GUI only
ros2 launch interaction_execution gui.launch.py

# GUI + all Python subsystems launched together
ros2 launch interaction_execution gui.launch.py \
  launch_perception:=true \
  launch_gamification:=true \
  launch_motion:=true
```

Note that for practical purposes besides testing subsystem interactivity, running SS3 as per the following is more atypical and more useful when running the whole system.

```bash
# GUI + voice control
ros2 launch interaction_execution gui.launch.py \
  launch_perception:=false \
  launch_gamification:=true \
  launch_motion:=false
```

#### Configuration

Pass launch arguments to `gui.launch.py`:

| Parameter | Default | Description |
|---|---|---|
| `auto_dispatch_motion` | false | If true, dispatch motion automatically once blocks are detected; if false, user must click button |
| `minimum_detected_blocks` | 1 | Minimum blocks required before transitioning to READY_TO_MOVE |
| `perception_timeout_s` | 20.0 | Max seconds to wait for perception gameboard; 0 = disabled |
| `max_scan_retries` | 1 | Number of automatic perception retries on timeout |
| `motion_timeout_s` | 20.0 | Max seconds to wait for motion completion; 0 = disabled |
| `launch_perception` | false | If true, launch perception node alongside GUI |
| `launch_gamification` | false | If true, launch gamification (solver) node alongside GUI |
| `launch_motion` | false | If true, launch motion planning node alongside GUI |
| `enable_voice_control` | true | If true, activate voice control interface |

**Example:**
```bash
ros2 launch interaction_execution gui.launch.py \
  perception_timeout_s:=25.0 \
  max_scan_retries:=2 \
  auto_dispatch_motion:=true
```

#### Troubleshooting

**GUI won't start**
Check Qt5 libraries are installed and ROS 2 is sourced:
```bash
sudo apt install qtbase5-dev qtwebengine5-dev
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

**Mission coordinator stuck in SCANNING**
- Verify perception node is running: `ros2 topic echo /perception/gameboard_state`
- If no messages, restart perception node and check for errors
- Increase `perception_timeout_s` if the system is slow

**Voice commands not recognised**
- Re-enroll voiceprint: `python3 voice_control/speaker_verification.py`
- Check microphone is connected and working: `arecord -l` (lists audio devices)
- Verify `enable_voice_control:=true` in launch arguments

**RViz view shows no blocks**
- Confirm `/perception/gameboard_state` is being published
- Check TF tree: `ros2 run tf2_tools view_frames`
- Verify robot driver is running and publishing `/tf`

**Wordle board not updating**
- Check `/gamification/guess` topic is being published: `ros2 topic echo /gamification/guess`
- Verify QWebEngineView has internet access (or cached Wordle JS file exists)
- Check browser console in GUI for JS errors (F12)

---



### Webcam alternative (no RealSense)

For testing CNN and gamification logic without the D435i. The webcam node publishes on the
same topic the perception node expects — no other code changes needed. Depth data will be
unavailable so 3D positions will be `null`, but letter classification and the Wordle solver
still work normally.

```bash
# Terminal 1 — start webcam publisher
python3 perception/src/webcam_node.py

# Terminal 2 — start perception node (picks up webcam feed automatically)
python3 perception/src/realsense_camera_cnn.py
```

> Set `DETECTION_MODE = "CARD"` in `realsense_camera_cnn.py` when using the webcam.
> The depth gate in `BLOCK` mode requires a RealSense depth stream.

---

## Training the CNN

### Step 0 — Print and prepare letter blocks

Print `perception/assets/block_labels.docx` and cut each label to **5 cm × 5 cm**.
Glue one label to the face of each wooden or 3D printed block (blocks should also be 5 cm × 5 cm).
Ensure the corner dot is visible in the top-left of each face — this is used for
orientation detection. Use a dark background (foam mat, dark desk) when scanning
so the white labels stand out clearly for the brightness threshold detector.

---

### Step 1 — Collect training data

```bash
python3 perception/src/collect_training_data.py
# Prompts to choose RealSense or webcam
# python3 collect_training_data.py realsense   ← skip prompt
# python3 collect_training_data.py webcam       ← skip prompt
```

- Press any **letter or digit key** to capture a burst of 4 images for that class.
- Move and tilt the card between every keypress — variation is essential.
- Watch the **Crop Preview** window — that is exactly what the CNN sees at runtime.
- Aim for **400–500 images per class** across multiple sessions and lighting conditions.
- The HUD shows a count per class; green = ≥ 200 images collected.
- Target: all 36 classes green before training.

### Step 2 — Train the model

```bash
cd perception/src
python3 train_letter_cnn.py
```

Key hyperparameters (top of `train_letter_cnn.py`):

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 1e-3 (AdamW + CosineAnnealing) |
| Val split | 15% |
| Test split | 10% |
| Label smoothing | 0.05 |
| Dropout | 0.4 |

The script produces two test evaluations:
- **Clean test** — no augmentation, measures learned accuracy.
- **Hard test** — rotation ±30°, lighting, blur, perspective distortion, occlusion. This predicts real-world camera performance.

Outputs saved to `perception/outputs/`:

| File | Description |
|---|---|
| `letter_cnn.pt` | Best model weights (saved at peak val accuracy) |
| `training_report.txt` | Per-class precision/recall for clean and hard test sets |
| `confusion_matrix.png` | Clean test confusion matrix |
| `confusion_matrix_hard.png` | Hard test confusion matrix |
| `confusion_matrix_epoch_N.png` | Validation snapshot every 5 epochs |
| `training_curves.png` | Training loss + validation accuracy curves |

Target before deployment: overall test accuracy ≥ 95%, no single class below 85% recall.

---

## Playing the game

### Mode A — Robot guesses

```bash
# 1. Select Mode A
ros2 topic pub --once /gamification/mode std_msgs/String "data: 'MODE_A'"

# 2. Set the secret word the robot will try to guess
ros2 topic pub --once /gamification/secret_word std_msgs/String "data: 'crane'"

# 3. Put letter blocks under the camera and press SPACEBAR in the perception window
#    The robot will automatically select and announce its first guess.

# 4. After the robot places the word, send feedback
#    G = correct position, B = wrong position, I = not in word
ros2 topic pub --once /gamification/feedback std_msgs/String "data: 'I I G B B'"

# Watch the current guess
ros2 topic echo /gamification/guess

# Watch full game state
ros2 topic echo /diagnostics
```

Correct startup order for Mode A: `MODE_A` → `secret_word` → `START` → scan blocks with SPACEBAR.

### Mode B — Human guesses

```bash
# 1. Select Mode B (robot picks a hidden word automatically)
ros2 topic pub --once /gamification/mode std_msgs/String "data: 'MODE_B'"

# 2. Submit your guess
ros2 topic pub --once /gamification/player_guess std_msgs/String "data: 'crane'"

# 3. Check your feedback
ros2 topic echo /diagnostics
# Look for "last_feedback": "GBIGG"
```

### Reset
```bash
ros2 topic pub --once /mission/state std_msgs/String "data: 'RESET'"
```

---

## ROS 2 topic reference

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/perception/detections` | String JSON | SS1 | Per-frame letter detections with camera-frame 3D positions |
| `/perception/gameboard_state` | GameboardState (latched) | SS1 | World-frame block poses — published once per completed scan |
| `/perception/human_detected` | Bool ~5 Hz | SS1 | Safety signal — triggers SAFETY_STOPPED and robot stop |
| `/perception/status` | String ~5 Hz | SS1 | SCANNING or IDLE |
| `/perception/image_annotated` | sensor_msgs/Image | SS1 | Annotated camera frame with detection overlays |
| `/mission/state` | String (latched) | SS3 | SCANNING / IDLE — controls perception scan window |
| `/wordle_bot/mission_state` | String (latched) | SS3 | 10-state coordinator FSM output |
| `/wordle_bot/mission_progress` | String JSON (latched) | SS3 | 7-step mission plan with per-step status for diagnostics |
| `/wordle_bot/start_mission` | Bool | SS3 | Pulse (5×) to start robot motion |
| `/wordle_bot/stop_mission` | Bool | SS3 | Emergency stop — published by safety guard on human detection |
| `/gamification/guess` | String | SS4 | Current 5-letter word guess |
| `/gamification/mission_state` | String JSON | SS4 | Ordered letter placement sequence with 3D pick positions |
| `/gamification/confirmation_status` | String JSON | SS4 | CONFIRMED / CONFIRMATION_FAILURE for letter verification phases |
| `/hl_control/word_request` | String (latched) | SS4 | Confirmed word — arms SS2 pick-and-place pipeline |
| `/diagnostics` | String JSON | SS4 | Full game state: status, attempt, candidates, found/looking-for letters |

---

## Configuration

### Detection tuning (top of `realsense_camera_cnn.py`)

| Variable | Default | Effect |
|---|---|---|
| `DETECTION_MODE` | `"CARD"` | `CARD` = white paper labels; `BLOCK` = bare wooden blocks with depth gate |
| `CNN_CONF_THRESHOLD` | `32.0` | Minimum confidence % to report a letter |
| `FRAMES_TO_AVERAGE` | `15` | Temporal smoothing window — CNN must agree across this many frames |
| `CARD_BRIGHTNESS` | `180` | Brightness threshold (0–255) for white paper detection |
| `CARD_MARGIN` | `0.10` | Fraction trimmed from each edge before CNN crop (removes paper border) |

### Camera intrinsics

Replace placeholder values with real values from your camera:
```bash
ros2 topic echo /camera/camera/color/camera_info --once
# K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
```
Then update `CAM_FX`, `CAM_FY`, `CAM_CX`, `CAM_CY` in `realsense_camera_cnn.py` (lines 110–113).

### Hand-eye calibration

Update `CAM_MOUNT_X/Y/Z` and `CAM_MOUNT_ROLL/PITCH/YAW` at the top of `realsense_camera_cnn.py`
with output from the ArUco hand-eye calibration procedure (ArUco ID 365, `handeye_realsense` package).
Edit the values and restart the node — no other code changes needed.

To verify calibration: place a block at a known world position, scan it, compare the published
world coordinates to the measured position. Error should be < 5 mm.

### Mission coordinator parameters (SS3)

Pass as launch arguments to `gui.launch.py`:

| Parameter | Default | Description |
|---|---|---|
| `auto_dispatch_motion` | false | Automatically dispatch motion when detections are ready |
| `minimum_detected_blocks` | 1 | Minimum blocks required before READY_TO_MOVE |
| `perception_timeout_s` | 0.0 | Scan timeout in seconds (0 = disabled) |
| `max_scan_retries` | 1 | Scan retry attempts before PERCEPTION_FAILED |
| `motion_timeout_s` | 20.0 | Motion completion timeout in seconds |

---

## Troubleshooting

**`WARNING: letter_cnn.pt not found`**
Run `python3 train_letter_cnn.py` first, or run `git lfs pull` if using Git LFS.
The node continues in fallback mode showing `?` for all detections.

**`[TF] camera→world lookup failed`**
The UR3e robot driver is not running. Start SS2 first. The perception node retries automatically — once the driver starts, TF lookups succeed without restart.

**`hl_control package not found`**
```bash
cd ~/ros2_ws && colcon build --packages-select hl_control && source install/setup.bash
```
Restart the perception node after building.

**Blocks detected but labelled `?`**
CNN confidence is below `CNN_CONF_THRESHOLD`. Check lighting, or temporarily lower the threshold to diagnose. If a specific class consistently fails, collect more training data for it and retrain.

**`gameboard_state` published with `frame_id = camera_color_optical_frame`**
TF was unavailable at scan completion — robot driver was not running. Restart the scan with the driver up.

**Gamification node shows `x_m: null` in placement JSON**
Depth data is missing. Confirm `align_depth.enable:=true` was passed to the RealSense launch and blocks are 10–80 cm from the camera.

**Gamification node receives no detections**
All five conditions must be true simultaneously: `game_active = True`, `game_mode = 'A'`, `mode_locked = True`, `guess_pending = True`, and the perception node must be AT POSITION with blocks visible. Correct order: `MODE_A` → `secret_word` → `START` → scan blocks.

**ROS 2 node list stalls**
```bash
ros2 daemon stop && ros2 daemon start && ros2 node list
```
Confirm `ROS_DOMAIN_ID=10` is set in every terminal.

**No RealSense devices found**
Check USB 3.0 connection (blue port). If you see `LIBUSB_ERROR_ACCESS`:
```bash
sudo chmod 666 /dev/bus/usb/*/*
```

---

## Licence

Developed as coursework for 41118 AI in Robotics at the University of Technology Sydney.
All rights reserved.
