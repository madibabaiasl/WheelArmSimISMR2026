# WheelArm-Sim: A Manipulation and Navigation Combined Multimodal Synthetic Data Generation Simulator for Unified Control in Assistive Robotics

**Authors:** Guangping Liu, Tipu Sultan, Vittorio Di Giorgio, Nick Hawkins, Flavio Esposito, Madi Babaiasl
**Venue:** 2026 
**Paper:** https://arxiv.org/abs/2601.21129 | **Video:** https://youtu.be/2Oy2y1wrFUo

---

## Get Started

### Requirements
Tested on:
- OS: Ubuntu 22.04 
- Nividia Issac Sim 4.2.0
- GPU: RTX A4000

### WheelArm-Sim
1. Download the simulator: /wheelarm_simulation.
2. Add the Data Collection as a GUI in your Issac Sim by extension.
3. Download the assets zip file and uncompress it in the directory(wheelarm_simulation/wheelarm_simulation/): https://sluedu-my.sharepoint.com/:u:/g/personal/guangping_liu_slu_edu/IQB43aNSs3LIRLAVYsyRot6iAfJuQWl1PN2k6Z4jJx5v5p0?e=QzPH3V
4. Install Kinova Gen3 ros2 package in your workspace: https://github.com/Kinovarobotics/ros2_kortex
5. Install manual_control_pkg as a package under the src of your kinova ros2 directory.

### Collect Data using WheelArm-Sim
1. Open the Example.usd in Issac Sim
2. Turn on the Data Collection GUI.
3. Start simulation.
4. Terminal1:
ros2 run manual_control_pkg data_processing_node
5. Terminal2:
ros2 run manual_control_pkg manual_control_node
6. Terminal3:
ros2 run manual_control_pkg inverse_kinematics_node
7. Terminal4:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
8. Fill task label and instructions in the Data Collection GUI
9. Start Collect data
10. Stop Collect data

### Sample Multimodal Dataset
Dataset:

