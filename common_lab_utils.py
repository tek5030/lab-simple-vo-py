import cv2
from dataclasses import dataclass
import numpy as np
from pylie import SO3, SE3
from anms import anms
from realsense_common import (CameraStream, CaptureMode)
from realsense_mono import (RealSenseSingleStreamCamera)


class Size:
    """Represents image size"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    @classmethod
    def from_numpy_shape(cls, shape):
        return cls(*shape[1::-1])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


@dataclass
class FrameToFrameCorrespondences:
    points_1: np.array([], dtype=np.float32)
    points_2: np.array([], dtype=np.float32)
    points_index_1: np.array([], dtype=np.float32)
    points_index_2: np.array([], dtype=np.float32)

    def size(self):
        return self.points_1.shape[0]


@dataclass
class MapToFrameCorrespondences:
    map_points: np.ndarray
    frame_points: np.ndarray
    map_point_indices: list
    frame_point_indices: list

    @property
    def size(self):
        return len(self.map_points)


def homogeneous(x):
    """Transforms Cartesian column vectors to homogeneous column vectors"""
    return np.r_[x, [np.ones(x.shape[1])]]


def hnormalized(x):
    """Transforms homogeneous column vector to Cartesian column vectors"""
    return x[:-1] / x[-1]


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self,
                 calibration_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray,
                 image_size: Size):
        """Constructs the camera model.

        :param calibration_matrix: The intrinsic calibration matrix.
        :param distortion_coeffs: Distortion coefficients on the form [k1, k2, p1, p2, k3].
        :param image_size: Size of image for this calibration.
        """
        self._calibration_matrix = calibration_matrix
        self._calibration_matrix_inv = np.linalg.inv(calibration_matrix)
        self._distortion_coeffs = distortion_coeffs
        self._image_size = image_size

    def undistort_image(self, distorted_image):
        """Undistorts an image corresponding to the camera model.

        :param distorted_image: The original, distorted image.
        :returns: The undistorted image.
        """

        return cv2.undistort(distorted_image, self._calibration_matrix, self._distortion_coeffs)

    def pixel_to_normalised(self, point_pixel):
        """Transform a pixel coordinate to normalised coordinates

        :param point_pixel: The 2D point in the image given in pixels.
        """

        if point_pixel.ndim == 1:
            # Convert to column vector.
            point_pixel = point_pixel[:, np.newaxis]

        return self._calibration_matrix_inv @ homogeneous(point_pixel)

    @property
    def calibration_matrix(self):
        """The intrinsic calibration matrix K."""
        return self._calibration_matrix

    @property
    def calibration_matrix_inv(self):
        """The inverse calibration matrix K^{-1}."""
        return self._calibration_matrix_inv

    @property
    def distortion_coeffs(self):
        """The distortion coefficients on the form [k1, k2, p1, p2, k3]."""
        return self._distortion_coeffs

    @property
    def image_size(self):
        """The image size"""
        return self._image_size

    @property
    def principal_point(self):
        """The principal point (p_u, p_v)"""
        return self._calibration_matrix[0, 2], self._calibration_matrix[1, 2]

    @property
    def focal_lengths(self):
        """The focal lengths (f_u, f_v)"""
        return self._calibration_matrix[0, 0], self._calibration_matrix[1, 1]

    @staticmethod
    def looks_at_pose(camera_pos_w: np.ndarray, target_pos_w: np.ndarray, up_vector_w: np.ndarray):
        """Computes the pose for a camera that looks at a given point."""
        cam_to_target_w = target_pos_w - camera_pos_w
        cam_z_w = cam_to_target_w.flatten() / np.linalg.norm(cam_to_target_w)

        cam_to_right_w = np.cross(-up_vector_w.flatten(), cam_z_w)
        cam_x_w = cam_to_right_w / np.linalg.norm(cam_to_target_w)

        cam_y_w = np.cross(cam_z_w, cam_x_w)

        return SE3((SO3(np.vstack((cam_x_w, cam_y_w, cam_z_w)).T), camera_pos_w))

    @staticmethod
    def jac_project_world_to_normalised_wrt_pose_w_c(pose_c_w: SE3, x_w: np.ndarray):
        """Computes the Jacobian for the projection of a world point to normalised coordinates wrt camera pose"""
        x_c = (pose_c_w * x_w).flatten()

        d = 1 / x_c[-1]
        xn = d * x_c

        return np.array([[-d, 0, d * xn[0], xn[0] * xn[1], -1 - xn[0] ** 2, xn[1]],
                         [0, -d, d * xn[1], 1 + xn[1] ** 2, -xn[0] * xn[1], -xn[0]]])

    @staticmethod
    def jac_project_normalised_wrt_x_c(x_c: np.ndarray):
        """Computes the Jacobian for the projection of a point in the camera coordinate system to normalised coordinates wrt the point"""
        x_c = x_c.flatten()
        d = 1 / x_c[-1]
        xn = d * x_c

        return np.array([[d, 0, -d * xn[0]],
                         [0, d, -d * xn[1]]])

    @classmethod
    def jac_project_world_to_normalised_wrt_x_w(cls, pose_c_w: SE3, x_w: np.ndarray):
        """Computes the Jacobian for the projection of a point in the world coordinate system to normalised coordinates wrt the point"""
        return cls.jac_project_normalised_wrt_x_c(pose_c_w * x_w) @ pose_c_w.jac_action_Xx_wrt_x()

    @staticmethod
    def project_to_normalised_3d(x_c: np.ndarray):
        """Projects a 3D point in the camera coordinate system onto the 3D normalised image plane"""
        return x_c / x_c[-1]

    @classmethod
    def project_to_normalised(cls, x_c: np.ndarray):
        """Projects a 3D point in the camera coordinate system onto the 2D normalised image plane"""
        xn = cls.project_to_normalised_3d(x_c)
        return xn[:2]

    @classmethod
    def reprojection_error_normalised(cls, x_c: np.ndarray, measured_x_n: np.ndarray):
        """Computes the reprojection error in normalised image coordinates"""
        return measured_x_n[:2] - cls.project_to_normalised(x_c)


class Frame:
    def __init__(self, image: np.ndarray, camera_model: PerspectiveCamera, keypoints: tuple, descriptors: np.array):
        self._gray_image = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._colour_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._image = image
        self._camera_model = camera_model
        self._keypoints = keypoints
        self._descriptors = descriptors
        self._pose_w_c = None

    @property
    def gray_image(self):
        return self._gray_image.copy()

    @property
    def colour_image(self):
        return self._colour_image.copy()

    @property
    def camera_model(self):
        return self._camera_model

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def descriptors(self):
        return self._descriptors

    @property
    def pose_w_c(self) -> SE3:
        return self._pose_w_c

    @pose_w_c.setter
    def pose_w_c(self, pose_w_c: SE3):
        self._pose_w_c = pose_w_c

    def has_pose(self) -> bool:
        return self._pose_w_c is not None


@dataclass
class Map:
    frame_1: Frame
    frame_2: Frame
    world_points: np.ndarray
    world_descriptors: np.ndarray

    @classmethod
    def create(cls, frame_1: Frame, frame_2: Frame, corr: FrameToFrameCorrespondences, world_points: np.ndarray):
        world_descriptors = frame_2.descriptors[corr.points_index_2, :]
        return cls(frame_1, frame_2, world_points, world_descriptors)


def extract_good_ratio_matches(matches, max_ratio):
    """
    Extracts a set of good matches according to the ratio test.

    :param matches: Input set of matches, the best and the second best match for each putative correspondence.
    :param max_ratio: Maximum acceptable ratio between the best and the next best match.
    :return: The set of matches that pass the ratio test.
    """
    if len(matches) == 0:
        return ()

    matches_arr = np.asarray(matches)
    distances = np.array([m.distance for m in matches_arr.ravel()]).reshape(matches_arr.shape)
    good = distances[:, 0] < distances[:, 1] * max_ratio

    # Return a tuple of good DMatch objects.
    return tuple(matches_arr[good, 0])


class CalibratedCamera:
    """Abstract base class for calibrated cameras

    Deliberately not using ABC, should we?
    """
    def __init__(self, camera_model: PerspectiveCamera):
        self._camera_model = camera_model

    @property
    def camera_model(self):
        return self._camera_model

    def capture_frame(self) -> np.ndarray:
        raise NotImplementedError("Called an abstract method in an abstract class!")

    def capture_undistorted_frame(self) -> np.ndarray:
        raise NotImplementedError("Called an abstract method in an abstract class!")


class CalibratedRealSenseCamera(CalibratedCamera):
    def __init__(self):
        self._cam = RealSenseSingleStreamCamera(CameraStream.LEFT)

        # Create model from calibration
        super().__init__(self._get_model_from_camera())

    def _get_model_from_camera(self) -> PerspectiveCamera:
        return PerspectiveCamera(
            self._cam.get_calibration_matrix(self._cam.active_stream),
            self._cam.get_distortion(self._cam.active_stream),
            self._cam.get_resolution(self._cam.active_stream)
        )

    def capture_frame(self) -> np.ndarray:
        raise NotImplementedError("CalibratedRealSenseCamera does not provide raw images in this lab!")

    def capture_undistorted_frame(self) -> np.ndarray:
        # NOTE: If not RECTIFIED, `_get_model_from_camera` will not be able to return data!
        self._cam.capture_mode = CaptureMode.RECTIFIED
        return self._cam.get_frame()


class CalibratedWebCamera(CalibratedCamera):
    def __init__(self, camera_model: PerspectiveCamera, video_source=0):
        # Create model from calibration
        super().__init__(camera_model)

        # Setup camera stream.
        self._cap = cv2.VideoCapture(video_source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_model.image_size.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_model.image_size.height)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source {video_source}")

    def capture_frame(self) -> np.ndarray:
        success, curr_frame = self._cap.read()

        if not success:
            raise RuntimeError(f"The video source stopped")

        return curr_frame

    def capture_undistorted_frame(self) -> np.ndarray:
        return self.camera_model.undistort_image(self.capture_frame())


class TrackingFrameExtractor:
    def __init__(self, camera: CalibratedCamera, detector, desc_extractor, use_anms=True):
        self._camera = camera
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._use_anms = use_anms

    def extract_frame(self) -> Frame:
        undist_frame = self._camera.capture_undistorted_frame()

        # Detect and describe features.
        gray_frame = undist_frame if undist_frame.ndim == 2 else cv2.cvtColor(undist_frame, cv2.COLOR_BGR2GRAY)
        keypoints = self._detector.detect(gray_frame)

        if len(keypoints) <= 5:
            return Frame(undist_frame, self._camera.camera_model, (), np.array([]))

        if self._use_anms:
            keypoints = self._adaptive_non_maximal_suppression(keypoints, self._camera.camera_model.image_size)

        keypoints, descriptors = self._desc_extractor.compute(gray_frame, keypoints)

        return Frame(undist_frame, self._camera.camera_model, keypoints, descriptors)

    @staticmethod
    def _adaptive_non_maximal_suppression(keypoints, img_size: Size, max_num=1000, max_ratio=0.7, tolerance=0.7):
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)

        num_to_retain = min(max_num, round(max_ratio * len(keypoints)))
        return anms.ssc(keypoints, num_to_retain, tolerance, img_size.width, img_size.height)


class Matcher:
    def __init__(self, norm_type: int, max_ratio: float = 0.8):
        self._matcher = cv2.BFMatcher_create(norm_type)
        self._max_ratio = max_ratio

    def match_frame_to_frame(self, frame_1: Frame, frame_2: Frame) -> FrameToFrameCorrespondences:
        matches = self._matcher.knnMatch(frame_2.descriptors, frame_1.descriptors, k=2)
        good_matches = extract_good_ratio_matches(matches, self._max_ratio)

        point_index_1 = [m.trainIdx for m in good_matches]
        point_index_2 = [m.queryIdx for m in good_matches]

        points_1 = [k.pt for k in np.asarray(frame_1.keypoints)[point_index_1]]
        points_2 = [k.pt for k in np.asarray(frame_2.keypoints)[point_index_2]]

        return FrameToFrameCorrespondences(
            np.asarray(points_1),
            np.asarray(points_2),
            np.asarray(point_index_1),
            np.asarray(point_index_2)
        )

    def match_map_to_frame(self, active_map: Map, frame: Frame) -> MapToFrameCorrespondences:
        matches = self._matcher.knnMatch(frame.descriptors, active_map.world_descriptors, k=2)
        good_matches = extract_good_ratio_matches(matches, self._max_ratio)

        map_ind = [m.trainIdx for m in good_matches]
        map_points = active_map.world_points[map_ind]

        frame_ind = [m.queryIdx for m in good_matches]
        frame_points = np.array([frame.keypoints[i].pt for i in frame_ind])

        return MapToFrameCorrespondences(map_points, frame_points, map_ind, frame_ind)
