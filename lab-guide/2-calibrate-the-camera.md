# Step 2: Calibrate the camera
To estimate the pose of the camera using correspondences, we will first need to specify the camera calibration parameters.

We have made a camera interface `CalibratedCamera`, which supports both RealSense cameras through the implementation `CalibratedRealSenseCamera`, and OpenCV cameras through `CalibratedWebCamera`.
These classes also contain the camera calibration.

You have the following choice of cameras:
- Use the precalibrated RealSense camera available at the lab with `CalibratedRealSenseCamera`.
  This is the recommended choice.
- Use a web camera available at the lab with `CalibratedWebCamera`.
  In this case, you can use the calibration results in [cameraParameters.xml](https://github.com/tek5030/lab-pose-estimation-py/blob/main/cameraParameters.xml).
  We have already filled in these data in the camera interface.
- Use your own camera with `CalibratedWebCamera`.
  In this case, you will have to calibrate the camera, and setup the parameters in `setup_camera_model_for_webcam()` at the bottom of [lab_simple_vo.py](../lab_simple_vo.py)

## 1. Choose the camera and, optionally, specify the camera calibration
Goto the bottom of [lab_simple_vo.py](../lab_simple_vo.py) and choose the camera you want to use.

Then, please continue to the [next step](3-finish-twoviewrelativeposeestimator.md).
