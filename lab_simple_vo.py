import cv2
import numpy as np
from common_lab_utils import (CalibratedCamera, CalibratedRealSenseCamera, CalibratedWebCamera)
from pose_estimators import (PnPPoseEstimator, MobaPoseEstimator)
from visualisation import (ArRenderer, Scene3D, print_info_in_image)


def run_simple_vo_lab(camera: CalibratedCamera):
    # Create the 2d-2d relative pose estimator.
    # We will use this for map creation.
    # FIXME: Implement!
    frame_to_frame_pose_estimator = TwoViewRelativePoseEstimator()

    # Create the 3d-2d pose estimator.
    # We will use this to navigate in maps between keyframes.
    init_pose_estimator = PnPPoseEstimator(camera.camera_model, do_iterative_estimation=False)
    pose_estimator = MobaPoseEstimator(init_pose_estimator, camera.camera_model)

    # Create points estimator.
    # We will use this estimator to triangulate points.
    # FIXME: Implement!
    init_points_estimator = DltPointsEstimator()
    points_estimator = SobaPointsEstimator(init_points_estimator)

    # Set up keypoint detector and descriptor extractor for correspondance matching.
    # FIXME: Own class for detection/extraction/matching? Add ANMS?
    detector = cv2.ORB_create(nfeatures=1000)
    desc_extractor = detector

    # Construct AR visualizer.
    # FIXME: Show world origin in camera view!
    ar_renderer = ArRenderer(camera.camera_model)

    # Construct 3D visualiser.
    # FIXME: Finish! Must add keyframes, current implementation just modified copy from pose estimation lab.
    scene_3d = Scene3D(camera.camera_model)

    # Construct empty references to hold frames and maps.
    tracking_frame = None
    active_keyframe = None
    init_map = None
    active_map = None

    # Main loop.
    while True:
        # Capture and make the frame ready for matching.
        # FIXME: Implement Frame
        tracking_frame = Frame(camera, detector, desc_extractor)

        # If we have an active map, track the frame using 2d-3d correspondences.
        if active_map is not None:
            # Compute 3d-2d correspondences.
            # FIXME: corr_2d_3d = matcher.match_map_to_frame(active_map, tracking_frame)

            # Estimate pose from 3d-2d correspondences.
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


class Frame:
    # FIXME: Implement!
    def __init__(self, camera: CalibratedCamera, detector, desc_extractor):
        pass


if __name__ == "__main__":
    # FIXME: Finish skeletons in common_lab_utils.py!
    # TODO 1: Choose camera.
    run_simple_vo_lab(CalibratedRealSenseCamera())
