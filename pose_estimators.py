import numpy as np
import cv2
from pylie import SO3, SE3
from dataclasses import dataclass
from common_lab_utils import (PerspectiveCamera)
from bundle_adjustment import (PrecalibratedCameraMeasurementsFixedWorld, PrecalibratedMotionOnlyBAObjective,
                               gauss_newton, levenberg_marquardt)


@dataclass
class PoseEstimate:
    """3D-2D pose estimation results"""
    pose_w_c: SE3 = None                        # Camera pose in the world.
    image_inlier_points: np.ndarray = None      # 2D inlier image points.
    world_inlier_points: np.ndarray = None      # 3D inlier world points.

    def is_found(self):
        """Checks if estimation succeeded.
        :return: True if result was found.
        """
        return self.pose_w_c is not None


class PnPPoseEstimator:
    """PnP-based pose estimator for calibrated camera with 2D-3D correspondences.

    This pose estimator first computes an initial result and extracts an inlier set using PnP.
    Then it optionally estimates the pose from the entire inlier set using an iterative method.
    """

    def __init__(self, camera_model: PerspectiveCamera, do_iterative_estimation=False):
        """Constructs the pose estimator.

        :param camera_model: The camera model for the calibrated camera.
        :param do_iterative_estimation: Estimates pose iteratively if True.
        """
        self._calibration_matrix = camera_model.calibration_matrix
        self._do_iterative_estimation = do_iterative_estimation

    def estimate(self, image_points, world_points):
        """Estimate camera pose from 2D-3D correspondences
        :param image_points: 2D image points in pixels.
        :param world_points: 3D world points.
        """

        # Check that we have a minimum required number of points, here 3 times the theoretic minimum.
        min_number_points = 9
        if len(image_points) < min_number_points:
            return PoseEstimate()

        # Find inliers and compute initial pose with RANSAC.
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, image_points, self._calibration_matrix, (),
                                                         useExtrinsicGuess=False, iterationsCount=10000,
                                                         reprojectionError=2.0, confidence=0.99,
                                                         flags=cv2.SOLVEPNP_AP3P)

        # Check that we have a valid result and enough inliers.
        if not retval or len(inliers) < min_number_points:
            return PoseEstimate()

        # Extract inliers.
        inliers = inliers.ravel()
        inlier_image_points = image_points[inliers]
        inlier_world_points = world_points[inliers]

        # Compute the camera pose with an iterative method using the entire inlier set.
        # Use "cv2.solvePnPRefineLM" on inlier points to improve "r_vec" and "t_vec".
        # Use the iterative method with current r_vec and t_vec as initial values.
        if self._do_iterative_estimation:
            rvec, tvec = cv2.solvePnPRefineLM(inlier_world_points, inlier_image_points,
                                              self._calibration_matrix, (), rvec, tvec)

        # We now have the pose of the world in the camera frame!
        pose_c_w = SE3((SO3.Exp(rvec), tvec))

        # Return the pose of the camera in the world frame.
        return PoseEstimate(pose_c_w.inverse(), inlier_image_points, inlier_world_points)


class MobaPoseEstimator:
    """Iterative pose estimator for calibrated camera with 2D-3D correspondences.
    This pose estimator needs another pose estimator, which it will use to initialise the estimate and find inliers.
    """
    def __init__(self, initial_pose_estimator, camera_model: PerspectiveCamera, print_cost=True):
        """Constructs pose estimator.
        :param initial_pose_estimator: Pointer to a pose estimator for initialiwation and inlier extraction.
        :param camera_model: Camera model
        """
        self._initial_pose_estimator = initial_pose_estimator
        self._camera_model = camera_model
        self._print_cost = print_cost

    def estimate(self, image_points, world_points):
        """Estimates camera pose from 2D-3D correspondences.
        :param image_points: 2D image points in pixels.
        :param world_points: 3D world points.
        """

        # Get initial pose estimate.
        estimate = self._initial_pose_estimator.estimate(image_points, world_points)

        if not estimate.is_found():
            return estimate

        # Create measurement set.
        measurement = PrecalibratedCameraMeasurementsFixedWorld(self._camera_model,
                                                                estimate.image_inlier_points,
                                                                estimate.world_inlier_points)

        # Create objective function.
        objective = PrecalibratedMotionOnlyBAObjective(measurement)

        # Optimize and update estimate.
        states, cost, _, _ = levenberg_marquardt(estimate.pose_w_c, objective)
        estimate.pose_w_c = states[-2]

        # Print cost.
        if self._print_cost:
            print(f"Motion only BA solved in {(len(cost) - 1):#2d} iterations, cost: {cost[0]:#f} -> {cost[-1]:#f}")

        return estimate

