import carla
import random
import weakref
import numpy as np


class SpawnManager:
    IMG_W = 1280
    IMG_H = 720

    def __init__(self, client: carla.Client, town: str = ""):
        self.client = client
        self.world  = client.load_world(town)
        self.bp_lib = self.world.get_blueprint_library()

        # Traffic Manager — must be synced too
        self.traffic_manager = client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.global_percentage_speed_difference(20)

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.vehicle = None
        self.rgb_cam = None
        self.seg_cam = None

        self.rgb_frame = None
        self.seg_frame = None
        self._rgb_ready = False
        self._seg_ready = False

        self._actors = []

    def spawn(self, vehicle_filter: str = "vehicle.tesla.model3"):
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in this map")

        random.shuffle(spawn_points)

        bp_list = self.bp_lib.filter(vehicle_filter)
        if not bp_list:
            raise RuntimeError(f"No blueprint matching: {vehicle_filter}")

        bp = random.choice(bp_list)
        bp.set_attribute("role_name", "hero")

        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(bp, sp)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("Could not spawn vehicle at any spawn point")

        self._actors.append(self.vehicle)

        # Autopilot via Traffic Manager
        self.vehicle.set_autopilot(True, 8000)

        # Camera transform
        cam_tf = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5)
        )

        # RGB camera
        rgb_bp = self.bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.IMG_W))
        rgb_bp.set_attribute("image_size_y", str(self.IMG_H))
        rgb_bp.set_attribute("fov", "90")

        self.rgb_cam = self.world.spawn_actor(rgb_bp, cam_tf, attach_to=self.vehicle)
        self._actors.append(self.rgb_cam)

        weak_self = weakref.ref(self)
        self.rgb_cam.listen(lambda img: SpawnManager._on_rgb(weak_self, img))

        # Segmentation camera
        seg_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(self.IMG_W))
        seg_bp.set_attribute("image_size_y", str(self.IMG_H))
        seg_bp.set_attribute("fov", "90")

        self.seg_cam = self.world.spawn_actor(seg_bp, cam_tf, attach_to=self.vehicle)
        self._actors.append(self.seg_cam)

        self.seg_cam.listen(lambda img: SpawnManager._on_seg(weak_self, img))

        # Warm-up
        print("[SpawnManager] Waiting for cameras...")
        for _ in range(20):
            self.world.tick()

        print(f"[SpawnManager] Spawned {vehicle_filter} at {sp.location}")

    # -------------------------
    # Traffic Light API
    # -------------------------

    def get_traffic_light(self):
        """Return traffic light affecting the ego vehicle (or None)."""
        if self.vehicle is None:
            return None
        return self.vehicle.get_traffic_light()

    def get_traffic_light_state(self):
        """Return CARLA traffic light state enum or None."""
        tl = self.get_traffic_light()
        if tl is None:
            return None
        return tl.get_state()

    def get_traffic_light_color(self):
        """
        Returns simplified string:
        'Red', 'Yellow', 'Green', or 'None'
        """
        tl = self.get_traffic_light()
        if tl is None:
            return "None"

        state = tl.get_state()

        if state == carla.TrafficLightState.Red:
            return "Red"
        elif state == carla.TrafficLightState.Yellow:
            return "Yellow"
        elif state == carla.TrafficLightState.Green:
            return "Green"
        else:
            return "None"

    # -------------------------
    # Frame handling
    # -------------------------

    def frames_ready(self) -> bool:
        return self._rgb_ready and self._seg_ready

    def pop_frames(self):
        self._rgb_ready = False
        self._seg_ready = False
        return self.rgb_frame.copy(), self.seg_frame.copy()

    # -------------------------
    # Cleanup
    # -------------------------

    def destroy(self):
        print("[SpawnManager] Destroying actors...")

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(False)

        for actor in reversed(self._actors):
            if actor is not None and actor.is_alive:
                actor.destroy()

        self._actors.clear()
        print("[SpawnManager] Done.")

    # -------------------------
    # Callbacks
    # -------------------------

    @staticmethod
    def _on_rgb(weak_self, image):
        self = weak_self()
        if self is None:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.IMG_H, self.IMG_W, 4))
        self.rgb_frame = arr[:, :, :3]
        self._rgb_ready = True

    @staticmethod
    def _on_seg(weak_self, image):
        self = weak_self()
        if self is None:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.IMG_H, self.IMG_W, 4))
        self.seg_frame = arr
        self._seg_ready = True