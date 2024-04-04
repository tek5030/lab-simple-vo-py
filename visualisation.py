import numpy as np
import pyvista as pv
import cv2
from pylie import SE3
from common_lab_utils import PerspectiveCamera, Frame
from estimators import PoseEstimate


class Scene3D:
    """Visualises the lab in 3D"""

    _do_exit = False
    _current_camera_actors = ()

    def __init__(self, camera_model: PerspectiveCamera):
        """Sets up the 3D viewer"""

        self._camera_model = camera_model
        self._plotter = pv.Plotter()

        self._do_exit = False
        self._current_camera_actors = ()
        self._keyframe_actors = ()
        self._point_cloud_actors = ()

        self._last_keyframe_position = None

        # Add callback for closing window.
        def exit_callback():
            self._do_exit = True
        self._plotter.add_key_event('q', exit_callback)

        # Add callback for printing current camera.
        def camera_callback():
            print(self._plotter.camera)
        self._plotter.add_key_event('c', camera_callback)

        # Set camera.
        self._plotter.camera.position = (1.15923, -1.11796, -5.18618)
        self._plotter.camera.up = (0.0184248, -0.977167, 0.211671)
        self._plotter.camera.focal_point = (0.398066, 0.0587764, 0.312404)

        # Show window.
        self._plotter.show(title="3D visualisation", interactive=True, interactive_update=True)

    def _update_current_camera_visualisation(self, frame: Frame):
        # Remove old visualisation.
        for actor in self._current_camera_actors:
            self._plotter.remove_actor(actor, render=False)

        # Render new visualisation.
        if frame.pose_w_c is not None:
            self._current_camera_actors = \
                add_frustum(self._plotter, frame.pose_w_c, self._camera_model, frame.colour_image) + \
                add_axis(self._plotter, frame.pose_w_c)

    def add_keyframe(self, frame: Frame):
        self._keyframe_actors += \
            add_frustum(self._plotter, frame.pose_w_c, frame.camera_model, frame.colour_image) + \
            add_axis(self._plotter, frame.pose_w_c)

        current_keyframe_position = frame.pose_w_c.translation

        if self._last_keyframe_position is not None:
            line = pv.Line(self._last_keyframe_position.ravel(), current_keyframe_position.ravel())

            self._keyframe_actors += \
                (self._plotter.add_mesh(line, color='b'), )

        self._last_keyframe_position = current_keyframe_position

    def add_point_cloud(self, pts_3d: np.ndarray):
        point_cloud = pv.PolyData(pts_3d)
        point_cloud['Map number'] = len(self._point_cloud_actors) * np.ones(len(pts_3d))

        self._point_cloud_actors += (self._plotter.add_mesh(point_cloud, render_points_as_spheres=True), )

    def reset(self):
        for actor in self._current_camera_actors:
            self._plotter.remove_actor(actor, render=False)
        self._current_camera_actors = ()

        for actor in self._keyframe_actors:
            self._plotter.remove_actor(actor, render=False)
        self._keyframe_actors = ()

        for actor in self._point_cloud_actors:
            self._plotter.remove_actor(actor, render=False)
        self._point_cloud_actors = ()

        self._last_keyframe_position = None

    def update(self, frame: Frame, time=10):
        """Updates the viewer with new camera frame"""

        self._update_current_camera_visualisation(frame)
        self._plotter.update(time)
        return self._do_exit


class ArRenderer:
    """Renders the 3D world scene in camera perspective"""

    def __init__(self, camera_model: PerspectiveCamera, hide_rendering=True):
        """Sets up the 3D viewer"""

        # Define tuple to hold keyframe actors.
        self._keyframe_actors = ()
        self._point_cloud_actors = ()

        # Set up plotter.
        # Set hide_rendering=False to show a window with the 3D rendering.
        theme = pv.themes.DefaultTheme()
        theme.transparent_background = True
        self._plotter = pv.Plotter(theme=theme, off_screen=hide_rendering)

        # Set camera pose.
        self._plotter.camera.position = (0., 0., 0.)
        self._plotter.camera.focal_point = (0., 0., 1.)
        self._plotter.camera.up = (0., -1., 0.)

        # Set principal point.
        p_u = camera_model.calibration_matrix[0, 2]
        p_v = camera_model.calibration_matrix[1, 2]
        img_width = camera_model.image_size.width
        img_height = camera_model.image_size.height
        self._plotter.camera.SetWindowCenter((-2 * p_u) / img_width + 1, (2 * p_v) / img_height - 1)

        # Set focal length.
        f_v = camera_model.calibration_matrix[1, 1]
        view_angle = 180.0 / np.pi * (2.0 * np.arctan2(img_height / 2.0, f_v))
        self._plotter.camera.view_angle = view_angle

        # Show window.
        self._plotter.show(title="AR visualisation", window_size=[img_width, img_height],
                           interactive=False, interactive_update=True)

    def add_keyframe(self, frame: Frame):
        self._keyframe_actors += add_axis(self._plotter, frame.pose_w_c)

    def set_current_point_cloud(self, pts_3d: np.ndarray):
        for actor in self._point_cloud_actors:
            self._plotter.remove_actor(actor, render=False)
        self._point_cloud_actors = ()

        point_cloud = pv.PolyData(pts_3d)
        point_cloud['Depth'] = pts_3d[:, -1]

        self._point_cloud_actors += (self._plotter.add_mesh(point_cloud, render_points_as_spheres=True), )
        self._plotter.remove_scalar_bar(render=False)

    def reset(self):
        for actor in self._keyframe_actors:
            self._plotter.remove_actor(actor, render=False)
        self._keyframe_actors = ()

        for actor in self._point_cloud_actors:
            self._plotter.remove_actor(actor, render=False)
        self._point_cloud_actors = ()

    def update(self, frame: Frame):
        """Updates the renderer with new camera pose estimate"""

        if not frame.has_pose():
            return None, None

        self._plotter.camera.model_transform_matrix = frame.pose_w_c.inverse().to_matrix()

        _, ar_rendering = self._plotter.show(return_cpos=True, return_img=True, screenshot=True,
                                             interactive_update=True)
        ar_rendering_bgr = ar_rendering[:, :, 2::-1]
        foreground_mask = ar_rendering[:, :, -1] > 0

        return ar_rendering_bgr, foreground_mask


def add_axis(plotter, pose: SE3, scale=0.1):
    """Adds a 3D axis object to the pyvista plotter"""

    T = pose.to_matrix()

    point = pv.Sphere(radius=0.1 * scale)
    point.transform(T)

    x_arrow = pv.Arrow(direction=(1.0, 0.0, 0.0), scale=scale)
    x_arrow.transform(T)

    y_arrow = pv.Arrow(direction=(0.0, 1.0, 0.0), scale=scale)
    y_arrow.transform(T)

    z_arrow = pv.Arrow(direction=(0.0, 0.0, 1.0), scale=scale)
    z_arrow.transform(T)

    axis_actors = (
        plotter.add_mesh(point),
        plotter.add_mesh(x_arrow, color='red', render=False),
        plotter.add_mesh(y_arrow, color='green', render=False),
        plotter.add_mesh(z_arrow, color='blue', render=False)
    )
    return axis_actors


def add_frustum(plotter, pose_w_c, camera_model, image, scale=0.1):
    """Adds a camera frustum to the pyvista plotter"""

    S = pose_w_c.to_matrix() @ np.diag([scale, scale, scale, 1.0])

    img_height, img_width = image.shape[:2]

    point_bottom_left = np.squeeze(camera_model.pixel_to_normalised(np.array([img_width-1., img_height-1.])))
    point_bottom_right = np.squeeze(camera_model.pixel_to_normalised(np.array([0., img_height-1.])))
    point_top_left = np.squeeze(camera_model.pixel_to_normalised(np.array([0., 0.])))
    point_top_right = np.squeeze(camera_model.pixel_to_normalised(np.array([img_width-1., 0.])))

    point_focal = np.zeros([3])

    pyramid = pv.Pyramid([point_bottom_left, point_bottom_right, point_top_left, point_top_right, point_focal])
    pyramid.transform(S)

    rectangle = pv.Rectangle([point_bottom_left, point_bottom_right, point_top_left])
    rectangle.texture_map_to_plane(inplace=True)
    rectangle.transform(S)

    image_flipped_rgb = image[::-1, :, ::-1].copy()
    tex = pv.numpy_to_texture(image_flipped_rgb)

    frustum_actors = (
        plotter.add_mesh(pyramid, show_edges=True, style='wireframe', render=False),
        plotter.add_mesh(rectangle, texture=tex, opacity=0.6, render=False)
    )
    return frustum_actors


def draw_detected_keypoints(vis_img, frame: Frame):
    for kp in frame.keypoints:
        cv2.drawMarker(vis_img, tuple(map(round, kp.pt)), (255, 0, 255), cv2.MARKER_CROSS, 5)


def draw_tracked_points(vis_img, points: np.ndarray):
    for pt in points.astype(np.int32):
        cv2.drawMarker(vis_img, pt, (255, 0, 255), cv2.MARKER_CROSS, 5)


def draw_two_view_matching(vis_img, correspondences):
    for pt1, pt2 in zip(correspondences.points_1.astype(np.int32), correspondences.points_2.astype(np.int32)):
        cv2.line(vis_img, pt1, pt2, (255, 0, 0))
        cv2.drawMarker(vis_img, pt1, (255, 255, 255), cv2.MARKER_CROSS, 5)
        cv2.drawMarker(vis_img, pt2, (255, 0, 255), cv2.MARKER_CROSS, 5)
