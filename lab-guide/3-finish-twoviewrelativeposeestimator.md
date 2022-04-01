# Step 3: Finish `TwoViewRelativePoseEstimator`

Go to `TwoViewRelativePoseEstimator` in [lab_simple_vo.py](../lab_simple_vo.py). 
Read through the code to get an overview. 

Study `TwoViewRelativePoseEstimator.estimate()` and try to understand what is happening here.

## 2. Use the 5-point algorithm to find an inlier set
Use the 5-point algorithm to find an inlier set that fits with a common epipolar geometry.

See [cv::findEssentialMat()](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f).

## 3. Compute the fundamental matrix based on the inlier set
Based on the inlier set from 2., compute the fundamental matrix using the 8-point algorithm.

Why shouldn't we use RANSAC here?

See [cv::findFundamentalMat()](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga30ccb52f4e726daa039fd5cb5bf0822b)

## 4. Compute the essential matrix from the fundamental matrix
Now that we have the fundamental matrix, compute the essential matrix using the camera calibration matrix.

## 5. Recover pose from the essential matrix
With the essential matrix, we are finally ready to estimate the relative pose between the two cameras. 
Decompose the essential matrix to recover the pose.

See [cv::recoverPose()](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0).

You should now be able to run and test the two-view pose estimator. 

Make sure the image window is in focus:
- Press \<space\> to do 2-view pose estimation and 3D reconstruction.
- Press \<r\> to reset.
- Press \<q\> to quit.

Play around with the two-view geometry!
Then continue to [the last page](4-finish-dltpointsestimator.md)
