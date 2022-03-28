import cv2
import numpy as np
from pylie import SO3, SE3


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


def retain_best(keypoints, num_to_keep):
    """Retains the given number of keypoints with highest response"""
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best


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


class CalibratedRealSenseCamera(CalibratedCamera):
    def __init__(self):
        # Create model from calibration
        super().__init__(self._get_model_from_camera())

    def _get_model_from_camera(self) -> PerspectiveCamera:
        # FIXME: Implement!
        pass

    def capture_frame(self):
        # FIXME: Implement!
        pass


class CalibratedWebCamera(CalibratedCamera):
    def __init__(self):
        # Create model from calibration
        super().__init__(self._get_model_from_calibration())

    def _get_model_from_calibration(self) -> PerspectiveCamera:
        # FIXME: Implement!
        # Insert parameters from calibration:
        pass

    def capture_frame(self):
        # FIXME: Implement!
        pass
