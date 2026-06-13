import cv2
import carla
from CarlaCodes.spawn_vec_cam import SpawnManager

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

manager = SpawnManager(client, town="Town01")

try:
    manager.spawn()
    while True:
        manager.world.tick()

        if not manager.frames_ready():
            continue

        rgb, seg = manager.pop_frames()
        cv2.imshow("RGB", rgb)
        # cv2.imshow("SEG", seg[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    manager.destroy()
    cv2.destroyAllWindows()