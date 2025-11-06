import open3d as o3d
import argparse
import os
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description="Open3D PLY viewer with screenshot, point size, and camera pose save/load")
    parser.add_argument("ply_path", type=str, help="Path to .ply point cloud file")
    args = parser.parse_args()

    if not os.path.isfile(args.ply_path):
        print(f"[✗] File not found: {args.ply_path}")
        return

    # Load point cloud
    pcd = o3d.io.read_point_cloud(args.ply_path)
    print(f"[✓] Loaded: {args.ply_path} ({np.asarray(pcd.points).shape[0]} points)")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Open3D Viewer",width=720,height=480)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    view_ctl = vis.get_view_control()

    frame_count = [0]
    point_size = [3.0]  # Initial point size
    settings_file = "view_settings.json"
    basename = os.path.splitext(os.path.basename(args.ply_path))[0]

    render_opt.point_size = point_size[0]

    # ---- Callback functions ----

    def update_point_size(vis_obj):
        render_opt.point_size = point_size[0]
        print(f"[~] Point size set to: {point_size[0]:.1f}")
        return False

    def increase_point_size(vis_obj):
        point_size[0] = min(point_size[0] + 0.1, 20.0)
        return update_point_size(vis_obj)

    def decrease_point_size(vis_obj):
        point_size[0] = max(point_size[0] - 0.1, 1.0)
        return update_point_size(vis_obj)

    def save_settings(vis_obj):
        cam = view_ctl.convert_to_pinhole_camera_parameters()
        settings = {
            "point_size": point_size[0],
            "camera": {
                "intrinsic": {
                    "width": cam.intrinsic.width,
                    "height": cam.intrinsic.height,
                    "fx": cam.intrinsic.get_focal_length()[0],
                    "fy": cam.intrinsic.get_focal_length()[1],
                    "cx": cam.intrinsic.get_principal_point()[0],
                    "cy": cam.intrinsic.get_principal_point()[1],
                },
                "extrinsic": cam.extrinsic.tolist()
            },
        }
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"[✓] Saved camera + point size to {settings_file}")
        return False

    def load_settings(vis_obj):
        if not os.path.isfile(settings_file):
            print(f"[!] {settings_file} not found")
            return False
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        point_size[0] = settings.get("point_size", point_size[0])
        update_point_size(vis_obj)

        # Reconstruct camera
        cam = o3d.camera.PinholeCameraParameters()
        intrinsic_data = settings["camera"]["intrinsic"]
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_data["width"],
            height=intrinsic_data["height"],
            fx=intrinsic_data["fx"],
            fy=intrinsic_data["fy"],
            cx=intrinsic_data["cx"],
            cy=intrinsic_data["cy"]
        )
        cam.extrinsic = np.array(settings["camera"]["extrinsic"])
        view_ctl.convert_from_pinhole_camera_parameters(cam)
        print(f"[✓] Loaded camera + point size from {settings_file}")
        return False

    def save_screenshot(vis_obj):
        filename = f"{basename}_screenshot_{frame_count[0]:03d}.png"
        vis_obj.capture_screen_image(filename)
        print(f"[✓] Screenshot saved: {filename}")
        frame_count[0] += 1
        return False

    # ---- Register key callbacks ----
    vis.register_key_callback(32, save_screenshot)     # SPACE
    vis.register_key_callback(264, increase_point_size)  # UP arrow
    vis.register_key_callback(265, decrease_point_size)  # DOWN arrow
    vis.register_key_callback(ord('S'), save_settings)
    vis.register_key_callback(ord('L'), load_settings)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
    
    
    
# topdown_camera_pose = np.array(
#       [[0.9905479967637421,0.13219405269931891,-0.036597794172154544,1.9164639767984166],
#       [-0.1332449029887764,0.8640013520179819,-0.4855383193308342,-0.3823859036870281],
#       [-0.032564734527541675,0.48582547909926377,0.8734489921701906,8.765351753327403],
#       [0.0,0.0,0.0,1.0]]
# )