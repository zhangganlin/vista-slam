import open3d as o3d
import numpy as np
import argparse
import os
from PIL import Image

def create_camera_lineset(scale=0.1, color=[0.6, 0.8, 1.0]):
    points = np.array([
        [0, 0, 0],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ], dtype=np.float64) * scale

    lines = [
        [0,1], [0,2], [0,3], [0,4],
        [1,2], [2,3], [3,4], [4,1]
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_camera_trajectory_lines(camera_centers, color=[0.0, 0, 1.0]):
    N = len(camera_centers)
    lines = [[i, i+1] for i in range(N-1)]
    colors = [color for _ in lines]

    traj_lineset = o3d.geometry.LineSet()
    traj_lineset.points = o3d.utility.Vector3dVector(camera_centers)
    traj_lineset.lines = o3d.utility.Vector2iVector(lines)
    traj_lineset.colors = o3d.utility.Vector3dVector(colors)
    return traj_lineset

def create_view_graph_lineset(view_graph, camera_centers):
    """
    view_graph: dict {int: list of int}
    camera_centers: np.array (N,3)
    Returns: o3d.geometry.LineSet connecting neighbors
    Colors edges blue if |vid1 - vid2| ≤ 3 else orange
    """
    view_ids = sorted(view_graph.keys())
    id_to_index = {vid: idx for idx, vid in enumerate(view_ids)}

    points = []
    for vid in view_ids:
        if vid < len(camera_centers):
            points.append(camera_centers[vid])
        else:
            points.append([0, 0, 0])
    points = np.array(points)

    lines = []
    colors = []
    for vid, neighbors in view_graph.items():
        if vid not in id_to_index:
            continue
        vid_idx = id_to_index[vid]
        for nb in neighbors:
            if nb in id_to_index:
                nb_idx = id_to_index[nb]
                # if vid_idx+1==nb_idx:
                #     lines.append([vid_idx, nb_idx])
                #     colors.append([0, 0, 1])  # blue

                if vid_idx < nb_idx:  # avoid duplicate edges
                    lines.append([vid_idx, nb_idx])
                    if abs(vid - nb) < 40 :
                        colors.append([0, 0, 1])  # blue
                    else:
                        colors.append([1, 0.5, 0])  # orange

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def make_background_transparent(png_path, output_path, bg_color=(0,0,0), threshold=10):
    img = Image.open(png_path).convert("RGBA")
    data = np.array(img)

    r = data[..., 0]
    g = data[..., 1]
    b = data[..., 2]
    a = data[..., 3]

    mask = (np.abs(r - bg_color[0]) < threshold) & \
           (np.abs(g - bg_color[1]) < threshold) & \
           (np.abs(b - bg_color[2]) < threshold)

    data[..., 3][mask] = 0  # set alpha to 0 (transparent)

    img2 = Image.fromarray(data)
    img2.save(output_path)
    print(f"[✓] Transparent PNG saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize PLY, camera poses (npy), and optional view graph (npz)")
    parser.add_argument("input_folder", type=str, help="Path to input folder containing .ply and .npy files, npz optional for view graph")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.input_folder, "pointcloud.ply")):
        print(f"PLY file not found: {os.path.join(args.input_folder, 'pointcloud.ply')}")
        return
    ply_path = os.path.join(args.input_folder, "pointcloud.ply")
    if not os.path.isfile(os.path.join(args.input_folder, "trajectory.npy")):
        print(f"Camera poses file not found: {os.path.join(args.input_folder, 'trajectory.npy')}")
        return
    poses_path = os.path.join(args.input_folder, "trajectory.npy")
    if not os.path.isfile(os.path.join(args.input_folder, "view_graph.npz")):
        print(f"View graph file not found: {os.path.join(args.input_folder, 'view_graph.npz')}, ignore edges in the visualization")
        view_graph_path = None
    else:
        view_graph_path = os.path.join(args.input_folder, "view_graph.npz")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded point cloud: {ply_path} with {len(pcd.points)} points")

    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4,4):
        print("Error: poses npy should have shape (N,4,4)")
        return
    print(f"Loaded {poses.shape[0]} camera poses from {poses_path}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud + Cameras + Trajectory + View Graph")
    vis.add_geometry(pcd)

    camera_centers = []
    for extrinsic in poses:
        lineset = create_camera_lineset(scale=0.1, color=[0.6, 0.8, 1.0])
        lineset.transform(extrinsic)
        vis.add_geometry(lineset)
        camera_centers.append(extrinsic[:3,3])
    camera_centers = np.array(camera_centers)

    traj_lineset = create_camera_trajectory_lines(camera_centers, color=[0.0, 0, 1.0])  # blue trajectory
    vis.add_geometry(traj_lineset)

    # Load and visualize view graph if provided
    if view_graph_path:
        view_graph_npz = np.load(view_graph_path, allow_pickle=True)
        if 'view_graph' not in view_graph_npz:
            print(f"Error: 'view_graph' not found in {view_graph_path}")
            return
        view_graph = view_graph_npz['view_graph'].item()
        view_graph_lineset = create_view_graph_lineset(view_graph, camera_centers)  # green edges
        vis.add_geometry(view_graph_lineset)
        print(f"Loaded and visualizing view graph from {view_graph_path}")

    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.line_width = 5.0

    frame_count = [0]

    def capture_transparent_screenshot(vis):
        opt = vis.get_render_option()
        orig_bg = opt.background_color

        # Set background to black before capture
        filename = f"{args.input_folder}/screenshot_{frame_count[0]:03d}_whitebg.png"
        vis.capture_screen_image(filename, do_render=True)
        print(f"[✓] Screenshot saved: {filename}")

        opt.background_color = np.array([0, 0, 0])
        filename = f"{args.input_folder}/screenshot_{frame_count[0]:03d}.png"
        vis.capture_screen_image(filename, do_render=True)
        print(f"[✓] Screenshot saved: {filename}")

        # Restore original background
        opt.background_color = orig_bg

        # Convert black background to transparent
        transparent_filename = f"{args.input_folder}/screenshot_{frame_count[0]:03d}_transparent.png"
        make_background_transparent(filename, transparent_filename)

        frame_count[0] += 1
        return False

    vis.register_key_callback(ord(' '), capture_transparent_screenshot)

    


    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()