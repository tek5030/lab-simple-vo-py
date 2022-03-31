# Step 1: Get an overview
At this stage, you should be able to get an overview of the project yourself.
The lab guide will therefore be a bit sparser from now on.

## Lab overview
The main steps in today's lab are:
- Create a 3D map from 2D-2D correspondences
  - Establish 2D-2D correspondences with keypoint feature matching
  - Estimate the fundamental matrix F
  - Compute the essential matrix E
  - Determine the relative pose from E
  - Triangulate world points using DLT and Structure-only Bundle Adjustment (SOBA)
- Estimate the camera pose from 2D-3D correspondences with the 3D map
  - Motion-only Bundle Adjustment (MOBA)
- Continue from the top

## Introduction to the project source files
To get an overview, please start by taking a look at [lab_simple_vo.py](../lab_simple_vo.py), and try to understand the steps.

Please continue to [the next step](2-calibrate-the-camera.md).