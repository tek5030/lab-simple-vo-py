import cv2
import numpy as np
from common_lab_utils import (CalibratedCamera, CalibratedRealSenseCamera, CalibratedWebCamera, Frame, Map, Size, PerspectiveCamera, TrackingFrameExtractor, Matcher, FrameToFrameCorrespondences)
from estimators import (PnPPoseEstimator, MobaPoseEstimator, PoseEstimate, SobaPointsEstimator, PointsEstimate, RelativePoseEstimate)
from pylie import SO3, SE3
from visualisation import (ArRenderer, Scene3D, print_info_in_image)


def run_simple_vo_lab(camera: CalibratedCamera):
    # Create the 2d-2d relative pose estimator.
    # We will use this for map creation.
    frame_to_frame_pose_estimator = TwoViewRelativePoseEstimator(camera.camera_model.calibration_matrix)

    # Create the 2d-3d pose estimator.
    # We will use this to navigate in maps between keyframes.
    init_pose_estimator = PnPPoseEstimator(camera.camera_model)
    pose_estimator = MobaPoseEstimator(init_pose_estimator, camera.camera_model)

    # Create points estimator.
    # We will use this estimator to triangulate points.
    points_estimator = DltPointsEstimator()
    # FIXME: Finish soba!
    # points_estimator = SobaPointsEstimator(init_points_estimator)

    # Set up keypoint detector and descriptor extractor for correspondence matching.
    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.ORB_create()
    frame_extractor = TrackingFrameExtractor(camera, detector, desc_extractor)
    matcher = Matcher(desc_extractor.defaultNorm())

    # Construct AR visualizer.
    # FIXME: Show world origin in camera view!
    ar_renderer = ArRenderer(camera.camera_model)

    # Construct 3D visualiser.
    # FIXME: Finish! Must add keyframes, current implementation just modified copy from pose estimation lab.
    scene_3d = Scene3D(camera.camera_model)

    # Construct empty references to hold frames and maps.
    active_keyframe = None
    active_map = None
    init_map = None

    # Main loop.
    while True:
        # Capture and make the frame ready for matching.
        tracking_frame = frame_extractor.extract_frame()

        # Construct image for visualisation.
        ar_frame = tracking_frame.colour_image

        # FIXME: Dummy estimate.
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
            estimate = frame_to_frame_pose_estimator.estimate(corr_2d_2d)

            if estimate.is_found():
                # Update frame poses with 2d-2d estimate (first camera is origin).
                tracking_frame.pose_w_c = estimate.pose_1_2

                # Compute an initial 3d map from the correspondences by using the epipolar geometry.
                init_estimate = points_estimator.estimate(active_keyframe, tracking_frame, estimate.inliers)
                init_map = Map.create(active_keyframe, tracking_frame, estimate.inliers, init_estimate.world_points)

                if init_map is not None:
                    # FIXME: Add 3d vis
                    # FIXME: Move to function in visualisation.py
                    for pt1, pt2 in zip(estimate.inliers.points_1.astype(np.int32),
                                        estimate.inliers.points_2.astype(np.int32)):
                        cv2.line(ar_frame, pt1, pt2, (255, 0, 0))
                        cv2.drawMarker(ar_frame, pt1, (255, 255, 255), cv2.MARKER_CROSS, 5)
                        cv2.drawMarker(ar_frame, pt2, (255, 0, 255), cv2.MARKER_CROSS, 5)

        # Update Augmented Reality visualization.
        ar_rendering, mask = ar_renderer.update(pose_estimate)
        if ar_rendering is not None:
            ar_frame[mask] = ar_rendering[mask]

        # FIXME: Stuff below is for preliminary testing.
        # FIXME: Add time, construct functions.
        if active_keyframe is None and active_map is None:
            for kp in tracking_frame.keypoints:
                cv2.drawMarker(ar_frame, tuple(map(round, kp.pt)), (255, 0, 255), cv2.MARKER_CROSS, 5)
        elif active_map is not None and pose_estimate.is_found():
            # Visualize tracking.
            # FIXME: Move to function in visualisation.py
            for pt in pose_estimate.image_inlier_points.astype(np.int32):
                cv2.drawMarker(ar_frame, pt, (255, 0, 255), cv2.MARKER_CROSS, 5)

        cv2.imshow("AR visualisation", ar_frame)
        key = cv2.waitKey(10)
        if key == ord('q'):
            print("Bye")
            break
        elif key == ord('r'):
            # Make all reference data empty.
            print("reset")
            active_keyframe = None
            active_map = None
            scene_3d.reset()
            ar_renderer.reset()
        elif key == ord(' '):
            if active_keyframe is None:
                print(f"set active keyframe")
                active_keyframe = tracking_frame

                active_keyframe.pose_w_c = SE3()
                scene_3d.add_keyframe(active_keyframe)
                ar_renderer.add_keyframe(active_keyframe)
            elif active_map is None:
                print(f"set active map")
                active_map = init_map
                scene_3d.add_point_cloud(active_map.world_points)
                ar_renderer.set_current_point_cloud(active_map.world_points)
                active_keyframe = active_map.frame_2
                scene_3d.add_keyframe(active_keyframe)
                ar_renderer.add_keyframe(active_keyframe)
            else:
                # Add a new consecutive map as an odometry step.
                if tracking_frame.pose_w_c is not None:
                    #  Use 2d-2d pose estimator to extract inliers for map point triangulation.
                    estimate = frame_to_frame_pose_estimator.estimate(
                        matcher.match_frame_to_frame(active_keyframe, tracking_frame)
                    )

                    # Try to create a new map based on the 2d-2d inliers.
                    points_estimate = points_estimator.estimate(active_keyframe, tracking_frame, estimate.inliers)
                    new_map = Map.create(active_keyframe, tracking_frame, estimate.inliers, points_estimate.world_points)

                    # FIXME: i cpp har man if (new_map), men den kan ikke returnere nullptr? jo, kanskje når det ikke er solution
                    if new_map is not None:
                        active_map = new_map
                        active_keyframe = tracking_frame

                        scene_3d.add_keyframe(active_keyframe)
                        scene_3d.add_point_cloud(active_map.world_points)
                        ar_renderer.add_keyframe(active_keyframe)
                        ar_renderer.set_current_point_cloud(active_map.world_points)
                    else:
                        print(f"--Map creation failed")

        do_exit = scene_3d.update(tracking_frame)
        if do_exit:
            break


class TwoViewRelativePoseEstimator:
    """Estimates the relative pose between to camera frames through epipolar geometry."""

    def __init__(self, K: np.ndarray, max_epipolar_distance: float = 1.0):
        """
        Constructor

        :param K: The intrinsic camera calibration matrix.
        """
        # FIXME: Gjør i henhold til resten av poseestimatorene
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
        if corr.size < min_number_points:
            return RelativePoseEstimate()

        # Get references to 2d-2d point correspondences
        points_1 = corr.points_1
        points_2 = corr.points_2

        # Find inliers with the 5-point algorithm
        p = 0.99
        _, mask = cv2.findEssentialMat(points_2, points_1, self._K, method=cv2.RANSAC, prob=p, threshold=self._max_epipolar_distance)
        mask = mask.ravel().astype(bool)

        # Extract inlier correspondences by using inlier mask.
        inlier_points_1 = points_1[mask]
        inlier_points_2 = points_2[mask]
        inlier_indices_1 = corr.points_index_1[mask]
        inlier_indices_2 = corr.points_index_2[mask]

        if inlier_points_1.shape[0] < min_number_points:
            return RelativePoseEstimate()

        # Compute Fundamental Matrix from all inliers.
        F, _ = cv2.findFundamentalMat(inlier_points_2, inlier_points_1, cv2.FM_8POINT)

        # Compute Essential Matrix from Fundamental matrix.
        E = self._K.T @ F @ self._K

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

    def estimate(self, frame_1: Frame, frame_2: Frame, corr):
        proj_mat_1 = frame_1.camera_model.calibration_matrix @ frame_1.pose_w_c.inverse().to_matrix()[:3, :]
        proj_mat_2 = frame_2.camera_model.calibration_matrix @ frame_2.pose_w_c.inverse().to_matrix()[:3, :]

        x_hom = cv2.triangulatePoints(proj_mat_1, proj_mat_2, corr.points_1.T, corr.points_2.T) # FIXME: AttributeError: 'tuple' object has no attribute 'points_1'
        x = x_hom[0:-1, :] / x_hom[-1, :]

        return PointsEstimate(x.T)

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
    # FIXME: Finish CalibratedRealSenseCamera in common_lab_utils.py!
    # TODO 1: Choose camera.
    video_source = 0
    run_simple_vo_lab(CalibratedWebCamera(setup_camera_model_for_webcam(), video_source))
