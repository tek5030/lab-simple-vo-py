import cv2
import numpy as np
from common_lab_utils import (CalibratedCamera, CalibratedRealSenseCamera, CalibratedWebCamera, Size, PerspectiveCamera, TrackingFrameExtractor)
from estimators import (PnPPoseEstimator, MobaPoseEstimator, PoseEstimate)
from visualisation import (ArRenderer, Scene3D, print_info_in_image)


def run_simple_vo_lab(camera: CalibratedCamera):
    # Create the 2d-2d relative pose estimator.
    # We will use this for map creation.
    # FIXME: Implement!
    frame_to_frame_pose_estimator = TwoViewRelativePoseEstimator()

    # Create the 2d-3d pose estimator.
    # We will use this to navigate in maps between keyframes.
    init_pose_estimator = PnPPoseEstimator(camera.camera_model, do_iterative_estimation=False)
    pose_estimator = MobaPoseEstimator(init_pose_estimator, camera.camera_model)

    # Create points estimator.
    # We will use this estimator to triangulate points.
    # FIXME: Implement!
    init_points_estimator = DltPointsEstimator()
    points_estimator = SobaPointsEstimator(init_points_estimator)

    # Set up keypoint detector and descriptor extractor for correspondence matching.
    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.ORB_create()
    frame_extractor = TrackingFrameExtractor(camera, detector, desc_extractor)

    # Construct AR visualizer.
    # FIXME: Show world origin in camera view!
    ar_renderer = ArRenderer(camera.camera_model)

    # Construct 3D visualiser.
    # FIXME: Finish! Must add keyframes, current implementation just modified copy from pose estimation lab.
    scene_3d = Scene3D(camera.camera_model)

    # Construct empty references to hold frames and maps.
    active_keyframe = None
    active_map = None

    # Main loop.
    while True:
        # Capture and make the frame ready for matching.
        tracking_frame = frame_extractor.extract_frame()

        # If we have an active map, track the frame using 2d-3d correspondences.
        if active_map is not None:
            # Compute 2d-3d correspondences.
            # FIXME: corr_2d_3d = matcher.match_map_to_frame(active_map, tracking_frame)

            # Estimate pose from 2d-3d correspondences.
            # FIXME: estimate = pose_estimator.estimate(corr_2d_3d.frame_points(), corr_2d_3d.map_points())
            pass

        # If we only have one active keyframe and no map,
        # visualise the map initialization from 2d-2d correspondences.
        elif active_keyframe is not None:
            # Compute 2d-2d correspondences.
            # FIXME: corr_2d_2d = matcher.match_frame_to_frame(active_keyframe, tracking_frame)

            # Estimate pose from 2d-2d correspondences.
            # FIXME: estimate = frame_to_frame_pose_estimator.estimate(corr_2d_2d)
            pass

        # FIXME: Dummy estimate.
        pose_estimate = PoseEstimate()

        # Update Augmented Reality visualization.
        ar_frame = tracking_frame.image
        ar_rendering, mask = ar_renderer.update(pose_estimate)
        if ar_rendering is not None:
            ar_frame[mask] = ar_rendering[mask]

        # FIXME: Stuff below is for preliminary testing.
        ar_frame = cv2.drawKeypoints(ar_frame, tracking_frame.keypoints, outImage=None, color=(0, 255, 0))

        cv2.imshow("AR visualisation", ar_frame)
        cv2.waitKey(10)

        do_exit = scene_3d.update(tracking_frame.image, pose_estimate)
        if do_exit:
            break


class TwoViewRelativePoseEstimator:
    # FIXME: Implement!
    def __init__(self):
        pass


class DltPointsEstimator:
    # FIXME: Implement!
    def __init__(self):
        pass


class SobaPointsEstimator:
    # FIXME: Implement!
    def __init__(self, init_estimator):
        pass


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
