# P4.8 final isolated environment

{
  "huggingface-hub": "0.35.3",
  "lerobot": "0.4.4",
  "mujoco": "3.3.6",
  "numpy": "2.2.6",
  "opencv-python": "4.12.0.88",
  "opencv-python-headless": "4.12.0.88",
  "robosuite": "1.4.0",
  "torch": "2.8.0+cu128",
  "torchcodec": "0.7.0",
  "torchvision": "0.23.0+cu128",
  "transformers": "4.57.6"
}

Official LeRobot8fff0fde source+metadata0.4.4; immutable simulator package/wheel hashes in install reports. Read-only inherited base Torch2.8+CUDA12.8. All new simulator dependencies and downgraded compatible transformer/Hub/OpenCV/MuJoCo versions live in the separate venv. Base package inventory remained identical. Missing libEGL dispatch library was extracted into an owned directory; system packages were not installed or upgraded. Use LD_LIBRARY_PATH=/root/autodl-tmp/p48_system_libs/usr/lib/x86_64-linux-gnu and MUJOCO_GL=egl.

Initial Mujoco3.17 joint-binding assertion and missing EGL failures are preserved. Actual smoke passes with Mujoco3.3.6. No model training.
