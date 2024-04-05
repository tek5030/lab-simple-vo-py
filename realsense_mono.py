import pyrealsense2 as rs2
import numpy as np
from realsense_common import (CameraIndex, CameraStream, CaptureMode, LaserMode, Size)


class RealSenseSingleStreamCamera:
    def __init__(self, active_stream: CameraStream = CameraStream.LEFT, capture_mode: CaptureMode = CaptureMode.RECTIFIED):
        connected_devices = rs2.context().devices
        if not connected_devices:
            raise RuntimeError(f"No RealSense device detected. Is it plugged in? Can you unplug and re-plug it?")
        for device in connected_devices:
            attrs = ['name', 'serial_number', 'firmware_version', 'usb_type_descriptor']
            name, serial_number, fw, usb = [device.get_info(getattr(rs2.camera_info, attr)) for attr in attrs]
            print(f"connected device: {name}, USB{usb} (S/N: {serial_number}, FW: {fw})")

        self._active_stream = active_stream
        self._pipe = rs2.pipeline()
        self._capture_mode = None
        self.capture_mode = capture_mode
        self.laser_mode = LaserMode.OFF

    def __del__(self):
        if hasattr(self, '_pipe'):
            self._pipe.stop()

    @property
    def active_stream(self):
        return self._active_stream

    @active_stream.setter
    def active_stream(self, active_stream):
        if active_stream == CameraStream.DEPTH:
            self.capture_mode = CaptureMode.RECTIFIED
        self._active_stream = active_stream

    @property
    def laser_mode(self):
        return

    @laser_mode.setter
    def laser_mode(self, laser_mode: LaserMode):
        depth_sensor = self._pipe.get_active_profile().get_device().first_depth_sensor()

        if not depth_sensor.supports(rs2.option.emitter_enabled):
            return

        if laser_mode is LaserMode.ON:
            depth_sensor.set_option(rs2.option.emitter_enabled, 1)
        else:
            depth_sensor.set_option(rs2.option.emitter_enabled, 0)

    @property
    def capture_mode(self):
        return self._capture_mode

    @capture_mode.setter
    def capture_mode(self, capture_mode: CaptureMode):
        if self._capture_mode == capture_mode:
            return

        self._capture_mode = capture_mode

        bgr_size = Size(width=848, height=480)  # or Size(width=640, height=480) ?
        if capture_mode is CaptureMode.RECTIFIED:
            mode = rs2.format.y8
            ir_size = bgr_size
        else:
            mode = rs2.format.y16
            ir_size = Size(width=1280, height=800)
            if self.active_stream == CameraStream.DEPTH:
                self.active_stream = CameraStream.LEFT

        try:
            self._pipe.stop()
        except RuntimeError:
            pass

        cfg = rs2.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs2.stream.infrared, int(CameraIndex.LEFT), **ir_size.dict, format=mode, framerate=0)
        cfg.enable_stream(rs2.stream.infrared, int(CameraIndex.RIGHT), **ir_size.dict, format=mode, framerate=0)
        cfg.enable_stream(rs2.stream.color, **bgr_size.dict, format=rs2.format.bgr8)
        if capture_mode is CaptureMode.RECTIFIED:
            cfg.enable_stream(rs2.stream.depth, format=rs2.format.z16)
        self._pipe.start(cfg)

        info = f"set {capture_mode}\n"
        info += f"  resolution: {self.get_resolution(self.active_stream)}\n"
        if self.capture_mode is not CaptureMode.UNRECTIFIED:
            info += f"  K: \n{self.get_calibration_matrix(self.active_stream)}\n"
        print(info)


    def get_frame(self):
        data = self._pipe.wait_for_frames()
        if self._active_stream == CameraStream.DEPTH:
            return np.asanyarray(data.get_depth_frame().get_data())
        elif self._active_stream == CameraStream.LEFT:
            return np.asanyarray(data.get_infrared_frame(CameraIndex.LEFT).get_data())
        elif self._active_stream == CameraStream.RIGHT:
            return np.asanyarray(data.get_infrared_frame(CameraIndex.RIGHT).get_data())

        return np.asanyarray(data.get_color_frame().get_data())

    def get_calibration_matrix(self, camera: CameraStream):
        i = self._get_video_stream_profile(camera).get_intrinsics()
        return np.array([
            [i.fx, 0, i.ppx],
            [0, i.fy, i.ppy],
            [0, 0, 1]
        ])

    def get_distortion(self, camera: CameraStream):
        d = self._get_video_stream_profile(camera).get_intrinsics().coeffs
        return np.array([d])

    def get_resolution(self, camera: CameraStream) -> Size:
        profile = self._get_video_stream_profile(camera)
        return Size(width=profile.width(), height=profile.height())

    def _get_video_stream_profile(self, camera: CameraStream):
        if camera == CameraStream.LEFT:
            return self._pipe.get_active_profile().get_stream(rs2.stream.infrared, CameraIndex.LEFT).as_video_stream_profile()
        elif camera == CameraStream.RIGHT:
            return self._pipe.get_active_profile().get_stream(rs2.stream.infrared, CameraIndex.RIGHT).as_video_stream_profile()
        elif camera == CameraStream.DEPTH:
            return self._pipe.get_active_profile().get_stream(rs2.stream.depth).as_video_stream_profile()
        return self._pipe.get_active_profile().get_stream(rs2.stream.color).as_video_stream_profile()
