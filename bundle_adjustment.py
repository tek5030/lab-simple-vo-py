import numpy as np
import scipy.linalg
from common_lab_utils import PerspectiveCamera
from pylie import SE3


class PrecalibratedCameraMeasurementsFixedWorld:
    """Measurements of fixed world points given in the normalised image plane"""

    def __init__(self, camera: PerspectiveCamera, u: np.ndarray, x_w: np.ndarray):
        """Constructs the 2D-3D measurement
        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param u: A 2xn matrix of n pixel observations.
        :param covs_u: A list of covariance matrices representing the uncertainty in each pixel observation.
        :param x_w: A 3xn matrix of the n corresponding world points.
        """

        self.camera = camera
        self.x_w = x_w.T

        # Transform to the normalised image plane.
        self.xn = camera.pixel_to_normalised(u.T)
        self.num = self.xn.shape[1]


class PrecalibratedCameraMeasurementsFixedCamera:
    """Measurements of world points given in the normalised image plane of a fixed camera"""

    def __init__(self, camera: PerspectiveCamera, pose_w_c: SE3, u: np.ndarray):
        """Constructs the 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param pose_w_c: The pose of the camera in the world frame.
        :param u: A 2xn matrix of n pixel observations corresponding to each and every 3D world point state.
        :param covs_u: A list of covariance matrices representing the uncertainty in each pixel observation.
        """

        self.camera = camera
        self.pose_w_c = pose_w_c
        self.pose_c_w = pose_w_c.inverse()

        # Transform to the normalised image plane.
        self.xn = camera.pixel_to_normalised(u)


class PrecalibratedMotionOnlyBAObjective:
    """Implements linearisation of motion-only BA objective function"""

    def __init__(self, measurement):
        """Constructs the objective
        :param measurement: A PrecalibratedCameraMeasurementsFixedWorld object.
        """
        self.measurement = measurement

    @staticmethod
    def extract_measurement_jacobian(point_index, pose_state_c_w, measurement):
        """Computes the measurement Jacobian for a specific point and camera measurement.
        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        A = measurement.camera.jac_project_world_to_normalised_wrt_pose_w_c(pose_state_c_w,
                                                                            measurement.x_w[:, [point_index]])

        return A

    @staticmethod
    def extract_measurement_error(point_index, pose_state_c_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.
        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.camera.reprojection_error_normalised(pose_state_c_w * measurement.x_w[:, [point_index]],
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, pose_state_w_c):
        """Linearises the objective over all states and measurements
        :param pose_state_w_c: The current camera pose state in the world frame.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_points = self.measurement.num

        A = np.zeros((2 * num_points, 6))
        b = np.zeros((2 * num_points, 1))

        pose_state_c_w = pose_state_w_c.inverse()

        for j in range(num_points):
            rows = slice(j * 2, (j + 1) * 2)
            A[rows, :] = self.extract_measurement_jacobian(j, pose_state_c_w, self.measurement)
            b[rows, :] = self.extract_measurement_error(j, pose_state_c_w, self.measurement)

        return A, b, b.T.dot(b)


class PrecalibratedStructureOnlyBAObjective:
    """Implements linearisation of the structure-only BA objective function"""

    def __init__(self, measurements):
        """Constructs the objective

        :param measurements: A list of PrecalibratedCameraMeasurementsFixedCamera objects, one for each camera.
        """
        self.measurements = measurements

    @staticmethod
    def extract_measurement_jacobian(point_index, point_state_w, measurement):
        """Computes the measurement Jacobian for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        A = measurement.camera.jac_project_world_to_normalised_wrt_x_w(measurement.pose_c_w, point_state_w)

        return A

    @staticmethod
    def extract_measurement_error(point_index, point_state_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.camera.reprojection_error_normalised(measurement.pose_c_w * point_state_w,
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, point_states_w):
        """Linearises the objective over all states and measurements

        :param point_states_w: The current state of the points in the world frame.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_cameras = len(self.measurements)
        num_points = len(point_states_w)

        A = np.zeros((2 * num_cameras * num_points, 3 * num_points))
        b = np.zeros((2 * num_cameras * num_points, 1))

        for i in range(num_cameras):
            for j in range(num_points):
                rows = slice(i * 2 * num_points + j * 2, i * 2 * num_points + (j + 1) * 2)
                cols = slice(j * 3, (j + 1) * 3)
                A[rows, cols] = self.extract_measurement_jacobian(j, point_states_w[j], self.measurements[i])
                b[rows, :] = self.extract_measurement_error(j, point_states_w[j], self.measurements[i])

        return A, b, b.T.dot(b)


class CompositeStateVariable:
    """A composition of a set of state variables.
    All variables must implement the __add__ and __sub__ operators appropriately,
    and return the dimension of the tangent space with the __len__ operator.
    """

    @property
    def variables(self):
        """The list of variables"""
        return self._variables

    @property
    def dim(self):
        """The dimension of the tangent space for the composited state manifold"""
        return self._dim

    def __init__(self, variables):
        """Constructs a composited state variable

        :param variables: A list of state variables.
        """
        self._variables = variables

        self._dim = 0
        for var in self.variables:
            self._dim = self._dim + len(var)

    def __add__(self, update_vector):
        """Add operator performs the "oplus" operation over all states.

        :param update_vector: A vector with the same dimension as self.dim().
        :return: The perturbed composite state
        """
        updated_variables = [None] * len(self.variables)
        start_elem = 0
        for i, current_state in enumerate(self.variables):
            end_elem = start_elem + len(current_state)
            updated_variables[i] = current_state + update_vector[start_elem:end_elem, :]
            start_elem = end_elem

        return CompositeStateVariable(updated_variables)

    def __getitem__(self, item):
        """Gets a specific state variable from the composite.

        :param item: An index in [0, len(self)).
        :return: The state
        """
        return self._variables[item]

    def __len__(self):
        """Returns the number of individual states in the composite

        :return: The number of states
        """
        return len(self._variables)

    def __sub__(self, other):
        """Subtract operator performs the "ominus" operation over all states.

        :param other: The other composite state
        :return: The difference vector in the composite tangent space
        """
        tau = np.zeros((self.dim, 1))
        start_elem = 0
        for i, (this_state, other_state) in enumerate(zip(self.variables, other.variables)):
            end_elem = start_elem + len(this_state)
            tau[start_elem:end_elem, :] = this_state - other_state
            start_elem = end_elem

        return tau


def gauss_newton(x_init, model, cost_thresh=1e-9, delta_thresh=1e-9, max_num_it=20):
    """Implements nonlinear least squares using the Gauss-Newton algorithm
    :param x_init: The initial state
    :param model: Model with a function linearise() that returns A, b and the cost for the current state estimate.
    :param cost_thresh: Threshold for cost function
    :param delta_thresh: Threshold for update vector
    :param max_num_it: Maximum number of iterations
    :return:
      - x: State estimates at each iteration, the final state in x[-1]
      - cost: The cost at each iteration
      - A: The full measurement Jacobian at the final state
      - b: The full measurement error at the final state
    """
    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    for it in range(max_num_it):
        A, b, cost[it] = model.linearise(x[it])
        tau = np.linalg.lstsq(A, b, rcond=None)[0]
        x[it + 1] = x[it] + tau

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    A, b, cost[-1] = model.linearise(x[-1])

    return x, cost, A, b


def levenberg_marquardt(x_init, model, cost_thresh=1e-9, delta_thresh=1e-9, max_num_it=20):
    """Implements nonlinear least squares using the Levenberg-Marquardt algorithm
    :param x_init: The initial state
    :param model: Model with a function linearise() that returns A, b and the cost for the current state estimate.
    :param cost_thresh: Threshold for cost function
    :param delta_thresh: Threshold for update vector
    :param max_num_it: Maximum number of iterations
    :return:
      - x: State estimates at each iteration, the final state in x[-1]
      - cost: The cost at each iteration
      - A: The full measurement Jacobian at the final state
      - b: The full measurement error at the final state
    """
    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    A, b, cost[0] = model.linearise(x[0])

    curr_lambda = 1e-4
    for it in range(max_num_it):
        inf_mat = A.T @ A

        tau = scipy.linalg.solve(inf_mat + np.diag(curr_lambda * np.diag(inf_mat)), A.T @ b, assume_a='pos')
        x_new = x[it] + tau

        A, b, cost_new = model.linearise(x_new)

        if cost_new < cost[it]:
            x[it + 1] = x_new
            cost[it + 1] = cost_new
            curr_lambda = 0.1 * curr_lambda
        else:
            x[it + 1] = x[it]
            cost[it + 1] = cost[it]
            curr_lambda = 10 * curr_lambda

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    return x, cost, A, b
