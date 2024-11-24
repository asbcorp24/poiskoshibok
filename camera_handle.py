import cv2
import threading

class CameraHandle:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraHandle, cls).__new__(cls)
                cls._instance.cam = None
                cls._instance._curr_camera_index = None
        return cls._instance

    def get_cam(self, camera_index):
        if self.cam is None or self._curr_camera_index != camera_index:
            self.cam, error = self.capture_global_camera(camera_index)
            if not error:
                self._curr_camera_index = camera_index
        return self.cam

    def capture_global_camera(self, camera_index):
        cam_num = int(camera_index.split()[-1]) if camera_index.split() else 0
        self.cam = cv2.VideoCapture(cam_num)
        if not self.cam.isOpened():
            print("Не удалось открыть камеру.")
            return None, True
        return self.cam, False

    def release_global_camera(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
            self._curr_camera_index = None

