# Lab: Simple Visual Odometry

This lab will be similar to [the pose estimation lab](https://github.com/tek5030/lab-pose-estimation-py), but we will now create our own 3D maps instead of relying on known planar world points.
This will also enable us to implement a very naive visual odometry method.

Start by cloning this repository on your machine.
Then open the lab project.

The lab is carried out by following these steps:
1. [Get an overview](lab-guide/1-get-an-overview.md)
2. [Calibrate the camera](lab-guide/2-calibrate-the-camera.md)
3. [Finish TwoViewRelativePoseEstimator](lab-guide/3-finish-twoviewrelativeposeestimator.md)
4. [Further experiments](lab-guide/4-finish-dltpointsestimator.md)

Please start the lab by going to the [first step](lab-guide/1-get-an-overview.md).

## Setup for Jetson (on the lab)
- Clone the repo into the directory `~/tek5030`
- Run the setup script `setup_jetson.bash` which 
  - creates a "venv"
  - installs a lot of stuff
  - downloads a precompiled VTK-wheel
  - download precompiled pyrealsense2
  - installs requirements from `requirements-jetson.txt`
- Open the editor of your choice

```bash
mkdir -p ~/tek5030
cd ~/tek5030
git clone https://github.com/tek5030/lab-simple-vo-py.git
cd lab-simple-vo-py
./setup_jetson.bash

# source venv/bin/activate
## connect camera
# python lab_simple_vo.py
```

:construction: If for some reason things doesn't work,
- try to run the script `./install_librealsense.bash`
- try to disconnect the camera and reconnect it
- `sudo reboot` the Jetson

If you see the error
> AttributeError: module 'pyrealsense2' has no attribute 'pipeline'

make sure to run `source venv/bin/activate` after `./setup_jetson.bash`.
