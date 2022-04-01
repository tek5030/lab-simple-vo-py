import cv2
import numpy as np
from common_lab_utils import (CalibratedCamera, CalibratedRealSenseCamera, CalibratedWebCamera, Frame, Map, Size,
                              PerspectiveCamera, TrackingFrameExtractor, Matcher, FrameToFrameCorrespondences,
                              homogeneous, hnormalized)
from estimators import (PnPPoseEstimator, MobaPoseEstimator, PoseEstimate,
                        SobaPointsEstimator, PointsEstimate, RelativePoseEstimate)
from pylie import SO3, SE3
from visualisation import (ArRenderer, Scene3D, draw_detected_keypoints, draw_tracked_points, draw_two_view_matching)


def run_simple_vo_solution(camera: CalibratedCamera):
    # Create the 2d-2d relative pose estimator.
    # We will use this for map creation.
    frame_to_frame_pose_estimator = TwoViewRelativePoseEstimator(camera.camera_model.calibration_matrix)

    # Create the 2d-3d pose estimator.
    # We will use this to navigate in maps between keyframes.
    init_pose_estimator = PnPPoseEstimator(camera.camera_model)
    pose_estimator = MobaPoseEstimator(init_pose_estimator, camera.camera_model)

    # Create points estimator.
    # We will use this estimator to triangulate points.
    init_points_estimator = DltPointsEstimator()
    points_estimator = SobaPointsEstimator(init_points_estimator)

    # Set up keypoint detector and descriptor extractor for correspondence matching.
    detector = cv2.ORB_create(nfeatures=2000, nlevels=16, scaleFactor=1.1,
                              scoreType=cv2.ORB_FAST_SCORE, fastThreshold=10)
    desc_extractor = detector
    frame_extractor = TrackingFrameExtractor(camera, detector, desc_extractor)
    matcher = Matcher(desc_extractor.defaultNorm())

    # Construct AR visualizer.
    ar_renderer = ArRenderer(camera.camera_model)

    # Construct 3D visualiser.
    scene_3d = Scene3D(camera.camera_model)

    # Construct empty references to hold frames and maps.
    active_keyframe = None
    active_map = None

    # Main loop.
    while True:
        # Capture and make the frame ready for matching.
        tracking_frame = frame_extractor.extract_frame()

        # Construct image for visualisation.
        ar_img = tracking_frame.colour_image

        # These will be used in visualisation below depending on the state.
        init_pose_estimate = RelativePoseEstimate()
        pose_estimate = PoseEstimate()

        # If we have an active map, track the frame using 2d-3d correspondences.
        if active_map is not None:
            # Compute 2d-3d correspondences.
            corr_2d_3d = matcher.match_map_to_frame(active_map, tracking_frame)

            # Estimate pose from 2d-3d correspondences.
            pose_estimate = pose_estimator.estimate(corr_2d_3d.frame_points, corr_2d_3d.map_points)

            if pose_estimate.is_found():
                # update frame pose with 2d-3d estimate.
                tracking_frame.pose_w_c = pose_estimate.pose_w_c

        # If we only have one active keyframe and no map,
        # visualise the map initialization from 2d-2d correspondences.
        elif active_keyframe is not None:
            # Compute 2d-2d correspondences.
            corr_2d_2d = matcher.match_frame_to_frame(active_keyframe, tracking_frame)

            # Estimate pose from 2d-2d correspondences.
            init_pose_estimate = frame_to_frame_pose_estimator.estimate(corr_2d_2d)

            if init_pose_estimate.is_found():
                # Update frame poses with 2d-2d estimate (first camera is origin).
                tracking_frame.pose_w_c = init_pose_estimate.pose_1_2

        # Update Augmented Reality visualization.
        ar_rendering, mask = ar_renderer.update(tracking_frame)
        if ar_rendering is not None:
            ar_img[mask] = ar_rendering[mask]

        # Visualise detections according to state.
        if active_keyframe is None and active_map is None:
            draw_detected_keypoints(ar_img, tracking_frame)

        elif active_map is not None and pose_estimate.is_found():
            draw_tracked_points(ar_img, pose_estimate.image_inlier_points)

        elif active_map is None and active_keyframe is not None and init_pose_estimate.is_found():
            draw_two_view_matching(ar_img, init_pose_estimate.inliers)

        # Display AR visualisation.
        cv2.imshow("AR visualisation", ar_img)
        key = cv2.waitKey(10)

        # Receive input from the keyboard.
        if key == ord('q'):
            # Exit
            print("Bye")
            break

        elif key == ord('r'):
            # Reset lab.
            print("reset")
            active_keyframe = None
            active_map = None
            scene_3d.reset()
            ar_renderer.reset()

        elif key == ord(' '):
            # Take action according to state.

            if active_keyframe is None:
                print("Setting first keyframe")
                active_keyframe = tracking_frame

                active_keyframe.pose_w_c = SE3()
                scene_3d.add_keyframe(active_keyframe)
                ar_renderer.add_keyframe(active_keyframe)

            elif active_map is None:
                print("Creating initial map")

                if not init_pose_estimate.is_found():
                    print("--Initial relative pose not found, no map created")
                    continue

                point_estimate = points_estimator.estimate(active_keyframe, tracking_frame, init_pose_estimate.inliers)

                if not point_estimate.is_found():
                    print("--Point triangulation failed, no map created")
                    continue

                active_map = Map.create(active_keyframe, tracking_frame, point_estimate.valid_correspondences, point_estimate.world_points)

                scene_3d.add_point_cloud(active_map.world_points)
                ar_renderer.set_current_point_cloud(active_map.world_points)
                active_keyframe = active_map.frame_2
                scene_3d.add_keyframe(active_keyframe)
                ar_renderer.add_keyframe(active_keyframe)

            else:
                print("Creating new map")

                if not tracking_frame.has_pose():
                    print("--Tracking frame pose not found, no map created")
                    continue

                # Add a new consecutive map as an odometry step.
                # Use 2d-2d pose estimator to extract inliers for map point triangulation.
                relative_pose_estimate = frame_to_frame_pose_estimator.estimate(
                    matcher.match_frame_to_frame(active_keyframe, tracking_frame)
                )

                if not relative_pose_estimate.is_found():
                    print("--Could estimate relative pose with keyframe, no map created")
                    continue

                # Try to create a new map based on the 2d-2d inliers.
                points_estimate = points_estimator.estimate(active_keyframe, tracking_frame, relative_pose_estimate.inliers)

                if not points_estimate.is_found():
                    print("--Could not triangulate points, no map created")
                    continue

                new_map = Map.create(active_keyframe, tracking_frame, points_estimate.valid_correspondences, points_estimate.world_points)

                active_map = new_map
                active_keyframe = tracking_frame

                scene_3d.add_keyframe(active_keyframe)
                scene_3d.add_point_cloud(active_map.world_points)
                ar_renderer.add_keyframe(active_keyframe)
                ar_renderer.set_current_point_cloud(active_map.world_points)

        do_exit = scene_3d.update(tracking_frame)
        if do_exit:
            break


class TwoViewRelativePoseEstimator:
    """Estimates the relative pose between to camera frames through epipolar geometry."""

    def __init__(self, K: np.ndarray, max_epipolar_distance=1.5):
        """
        Constructor

        :param K: The intrinsic camera calibration matrix.
        """
        self._K = K
        self._max_epipolar_distance = max_epipolar_distance

    def estimate(self, corr: FrameToFrameCorrespondences) -> RelativePoseEstimate:
        """
        Estimate the relative pose from 2d-2d correspondences.

        :param corr: The 2d-2d correspondences between two frames.
        """
        # Set a minimum required number of points,
        # here 3 times the theoretical minimum.
        min_number_points = 3 * 5

        # Check that we have enough points.
        if corr.size() < min_number_points:
            return RelativePoseEstimate()

        # Get references to 2d-2d point correspondences
        points_1 = corr.points_1
        points_2 = corr.points_2

        # TODO 2: Use cv2.findEssentialMat() to get an inlier mask for the correspondences. Use self._max_epipolar_distance!
        # Find inliers with the 5-point algorithm
        _, mask = cv2.findEssentialMat(points_2, points_1, self._K, method=cv2.RANSAC,
                                       threshold=self._max_epipolar_distance)
        mask = mask.ravel().astype(bool)

        # Extract inlier correspondences by using inlier mask.
        inlier_points_1 = points_1[mask]
        inlier_points_2 = points_2[mask]
        inlier_indices_1 = corr.points_index_1[mask]
        inlier_indices_2 = corr.points_index_2[mask]

        if inlier_points_1.shape[0] < min_number_points:
            return RelativePoseEstimate()

        # TODO 3: Compute the fundamental matrix from the entire inlier set using cv2.findFundamentalMat()
        # Compute Fundamental Matrix from all inliers.
        F, _ = cv2.findFundamentalMat(inlier_points_2, inlier_points_1, cv2.FM_8POINT)

        # TODO 4: Compute the essential matrix from the fundamental matrix.
        # Compute Essential Matrix from Fundamental matrix.
        E = self._K.T @ F @ self._K

        # TODO 5: Estimate pose from the essential matrix with cv2.recoverPose().
        # Recover pose from Essential Matrix.
        num_pass_check, R, t, mask = cv2.recoverPose(E, inlier_points_2, inlier_points_1, self._K)

        if num_pass_check < min_number_points:
            return RelativePoseEstimate()

        mask = mask.ravel().astype(bool)

        # Extract inlier correspondences that pass the cheirality check.
        inlier_points_1 = inlier_points_1[mask]
        inlier_points_2 = inlier_points_2[mask]
        inlier_indices_1 = inlier_indices_1[mask]
        inlier_indices_2 = inlier_indices_2[mask]

        # Return estimate.
        inlier_corr = FrameToFrameCorrespondences(
            np.asarray(inlier_points_1),
            np.asarray(inlier_points_2),
            np.asarray(inlier_indices_1),
            np.asarray(inlier_indices_2)
        )

        return RelativePoseEstimate(SE3((SO3(R), t)), inlier_corr, num_pass_check)


class DltPointsEstimator:
    """Points estimator based on the DLT triangulation algorithm."""
    def __init__(self, min_disparity=5.):
        self._min_disparity = min_disparity

    def _compute_disparity_mask(self, frame_1: Frame, frame_2: Frame, corr):
        # TODO 6: Compute disparities in the general two-view case.
        R_1_2 = (frame_1.pose_w_c.rotation.inverse() @ frame_2.pose_w_c.rotation).matrix

        H_1_2 = frame_1.camera_model.calibration_matrix @ \
                R_1_2 @ \
                np.linalg.inv(frame_2.camera_model.calibration_matrix)

        u_1_inf = hnormalized(H_1_2 @ homogeneous(corr.points_2.T))
        disparities = np.linalg.norm(corr.points_1.T - u_1_inf, axis=0)

        return disparities >= self._min_disparity

    def estimate(self, frame_1: Frame, frame_2: Frame, corr):
        valid_mask = self._compute_disparity_mask(frame_1, frame_2, corr)

        valid_corr = FrameToFrameCorrespondences(
            corr.points_1[valid_mask],
            corr.points_2[valid_mask],
            corr.points_index_1[valid_mask],
            corr.points_index_2[valid_mask]
        )

        # TODO 7: Triangulate points.
        proj_mat_1 = frame_1.camera_model.calibration_matrix @ frame_1.pose_w_c.inverse().to_matrix()[:3, :]
        proj_mat_2 = frame_2.camera_model.calibration_matrix @ frame_2.pose_w_c.inverse().to_matrix()[:3, :]

        x_hom = cv2.triangulatePoints(proj_mat_1, proj_mat_2, valid_corr.points_1.T, valid_corr.points_2.T)
        x = hnormalized(x_hom)

        return PointsEstimate(x.T, valid_corr, valid_mask)


def setup_camera_model_for_webcam():
    """Constructs the camera model according to the results from camera calibration"""

    # TODO 1.1: Set K according to calibration.
    # Set calibration matrix K
    K = np.array([
        [6.6051081297156020e+02, 0., 3.1810845757653777e+02],
        [0., 6.6051081297156020e+02, 2.3995332228230293e+02],
        [0., 0., 1.]
    ])

    # TODO 1.2: Set dist_coeffs according to the calibration.
    dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])

    # TODO 1.3: Set the image size corresponding to the calibration
    image_size = Size(640, 480)

    return PerspectiveCamera(K, dist_coeffs, image_size)


if __name__ == "__main__":
    # TODO 1: Choose camera.

    # RealSense
    run_simple_vo_solution(CalibratedRealSenseCamera())

    # Webcam
    # video_source = 0
    # run_simple_vo_solution(CalibratedWebCamera(setup_camera_model_for_webcam(), video_source))
