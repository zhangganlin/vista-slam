import open3d as o3d
import numpy as np
import glob
import cv2
from scipy.spatial.transform import Rotation as R
from glob import glob
import os
import json


input_prefix = "output"
input_folder = f"{input_prefix}/test"

# output_folder = f"apt0_video_frames_look_forward"
output_folder = f"{input_folder}/topdown"
os.makedirs(output_folder, exist_ok=True)


topdown = True
pose_offset = np.eye(4)
if topdown==False:
    pose_offset[:3,:3] = R.from_rotvec([1,0,0],20).as_matrix()
    pose_offset[:3,3] = [0,0.3,5]
# topdown_camera_pose = np.array(
#         [[-9.39070625e-01, -3.42904054e-01, -2.37312198e-02,  5.29013290e+00],
#         [-3.42857276e-01,  9.39367441e-01, -6.13987154e-03, -2.88170663e+00],
#         [ 2.43977221e-02,  2.37064840e-03, -9.99699520e-01,  8.65166061e+00],
#         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],])

#bf_ap0
topdown_camera_pose = np.array(
      [[0.9909692247271552,0.11974371453687545,-0.06034433256381569,2.3765190701497647],
      [-0.1290855380232213,0.7301299165747327,-0.67100464141154,-0.12965902817085873],
      [-0.03628938573352524,0.6727345299235689,0.7389934591940569,8.543355929727054],
      [0.0,0.0,0.0,1.0]]
)

settings_file = "view_settings.json"
if os.path.isfile(settings_file):
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    topdown_camera_pose = np.array(settings["camera"]["extrinsic"])
    

    
camera_color = [0.6, 0.8, 1.0]
cur_camera_color = [0.0745, 0.3137,0.1059]
pre_camera_color = [0.0431, 0.4627, 0.6274]
background_color = [1.0, 1.0, 1.0] # white background

scale_paths = glob(f"{input_folder}/scales_*.npy")
pgo_idx = sorted([int(p.split("_")[-1].split(".")[0])  for p in scale_paths])

length = pgo_idx[-1]+1

depths = np.load(f"{input_folder}/depths.npy")
images = (np.load(f"{input_folder}/images.npy")*255).astype(np.uint8)
intrinsics = np.load(f"{input_folder}/intrinsics.npy")
confs_npz = np.load(f"{input_folder}/confs.npz",allow_pickle=True)
confs = confs_npz['confs']
conf_thres = confs_npz['thres']
masks = confs>conf_thres

view_graph_npz = np.load(f"{input_folder}/view_graph.npz",allow_pickle=True)
view_graph = view_graph_npz['view_graph'].item()
view_names = view_graph_npz['view_names'].tolist()
loop_min_dist = view_graph_npz['loop_min_dist'].item()


pgo_i = 0
traj = np.load(f"{input_folder}/trajectory_{pgo_idx[pgo_i]}.npy")
scales = np.load(f"{input_folder}/scales_{pgo_idx[pgo_i]}.npy")




def create_pointcloud_from_rgbd(depth, color, intrinsic, pose, mask=None,
                                depth_scale=1000.0, depth_trunc=20000.0):
    """
    depth: HxW numpy array (depth in same unit as depth_scale)
    color: HxWx3 numpy array (uint8 RGB)
    intrinsic: (3, 3) numpy array
    pose: (4, 4) numpy array (camera_to_world)
    mask: HxW boolean numpy array (True = keep, False = remove)
    """
    # Convert numpy arrays to Open3D images
    depth_o3d = o3d.geometry.Image((depth*1000).astype(np.uint16))
    color_o3d = o3d.geometry.Image(color.astype(np.uint8))

    # Make an RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # Convert intrinsic to PinholeCameraIntrinsic
    h, w = depth.shape
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    # Create point cloud in camera coordinates
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
    # Transform to world coordinates
    pcd.transform(pose)

    if mask is not None:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask_flat = mask.flatten()
        points = points[mask_flat]
        colors = colors[mask_flat]/1.5
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def set_camera_params(vis,extrinsic=None, offset=True):
    ctr = vis.get_view_control()
    camera = ctr.convert_to_pinhole_camera_parameters()

    #0000
    if topdown:
        extrinsic = topdown_camera_pose
    if offset:
        extrinsic = np.matmul(pose_offset,extrinsic)

    camera.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera)
    vis.update_renderer()

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(visible = False, width=720,height=480)
render_option = vis.get_render_option()
render_option.point_size = 1.0
render_option.line_width = 0.8
render_option.background_color = background_color

def add_view(vis, i, add_cam=True):
    pose = traj[i]
    depth = depths[i] * scales[i]
    color = images[i]
    mask = masks[i]
    intri = intrinsics[i]
    points = create_pointcloud_from_rgbd(depth,color,intri,pose,mask)
    vis.add_geometry(points)
    est_w2c = np.linalg.inv(pose)
    if add_cam:
        small_cur_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                    intrinsic=intri, extrinsic=est_w2c, scale=0.1)
        small_cur_cam.lines = o3d.utility.Vector2iVector(np.asarray(small_cur_cam.lines)[:8])
        small_cur_cam.colors = o3d.utility.Vector3dVector(np.asarray(small_cur_cam.colors)[:8])
        small_cur_cam.colors = o3d.utility.Vector3dVector([camera_color for _ in range(8)])    
        vis.add_geometry(small_cur_cam)
    
    edge_lineset = get_edge_lineset(i)
    vis.add_geometry(edge_lineset)

    
def get_edge_lineset(v):
    neighbor_edge_color = [0,0,1.0]
    loop_edge_color = [1,0.5,0]
    neighbor_views = view_graph[v]
    v_center = traj[v][:3,3]
    points = [v_center]    
    lines = []
    colors = []
    point_idx = 0
    for nv in neighbor_views:
        if nv > v:
            continue
        point_idx += 1
        nv_center = traj[nv][:3,3]
        points.append(nv_center)
        lines.append([0,point_idx])
        if abs(v-nv) >= loop_min_dist:
            colors.append(loop_edge_color)
        else:
            colors.append(neighbor_edge_color)
    if len(lines) > 0:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    return None

cur_est_cam = None
last_w2c = None
vis_pre_cam = None
for v in range(length):
# for v in range(9,11):
    # vis.clear_geometries()
    assert v <= pgo_idx[pgo_i]
    pose = traj[v]
    depth = depths[v] * scales[v]
    color = images[v]
    mask = masks[v]
    intri = intrinsics[v]
    points = create_pointcloud_from_rgbd(depth,color,intri,pose,mask)

    vis.add_geometry(points)
        
    est_w2c = np.linalg.inv(pose)
    if last_w2c is None:
        last_w2c = est_w2c
    
    vis_cur_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                   intrinsic=intri, extrinsic=est_w2c, scale=1.0)
    vis_cur_cam.lines = o3d.utility.Vector2iVector(np.asarray(vis_cur_cam.lines)[:8])
    vis_cur_cam.colors = o3d.utility.Vector3dVector([cur_camera_color for _ in range(8)])
    
    vis.add_geometry(vis_cur_cam)

    
    edge_lineset = get_edge_lineset(v)
    if edge_lineset is not None:
        vis.add_geometry(edge_lineset)
        
    set_camera_params(vis,est_w2c)
    
    rendered_img = vis.capture_screen_float_buffer(True)
    img_np = np.asarray(rendered_img)
    img_np = img_np[...,[2,1,0]]
    cv2.imwrite(f"{output_folder}/{v:04d}.png",img_np*255)
        
    
    if v == pgo_idx[pgo_i]:
        
        if pgo_i== len(pgo_idx)-1:
            traj = np.load(f"{input_folder}/trajectory.npy")
            scales = np.load(f"{input_folder}/scales.npy")
        else:    
            pgo_i += 1
            traj = np.load(f"{input_folder}/trajectory_{pgo_idx[pgo_i]}.npy")
            scales = np.load(f"{input_folder}/scales_{pgo_idx[pgo_i]}.npy")
        vis.clear_geometries()
        for i in range(v+1):
            if i == v-1 or i==v:
                add_view(vis,i,False)
            else:
                add_view(vis,i,True)

        vis_pre_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                    intrinsic=intri, extrinsic=last_w2c, scale=0.8)
        vis_pre_cam.lines = o3d.utility.Vector2iVector(np.asarray(vis_pre_cam.lines)[:8])
        vis_pre_cam.colors = o3d.utility.Vector3dVector(np.asarray(vis_pre_cam.colors)[:8])
        vis_pre_cam.colors = o3d.utility.Vector3dVector([pre_camera_color for _ in range(8)])
        vis.add_geometry(vis_pre_cam)    
            
        vis_cur_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                intrinsic=intri, extrinsic=est_w2c, scale=1.0)
        vis_cur_cam.lines = o3d.utility.Vector2iVector(np.asarray(vis_cur_cam.lines)[:8])
        vis_cur_cam.colors = o3d.utility.Vector3dVector([cur_camera_color for _ in range(8)])
        vis.add_geometry(vis_cur_cam)
        set_camera_params(vis,est_w2c)
        
        rendered_img = vis.capture_screen_float_buffer(True)
        img_np = np.asarray(rendered_img)
        img_np = img_np[...,[2,1,0]]
        cv2.imwrite(f"{output_folder}/{v:04d}_pgo.png",img_np*255)
    
    if vis_pre_cam is not None:
        vis.remove_geometry(vis_pre_cam)
        small_prev_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                    intrinsic=intri, extrinsic=last_w2c, scale=0.1)
        small_prev_cam.lines = o3d.utility.Vector2iVector(np.asarray(small_prev_cam.lines)[:8])
        small_prev_cam.colors = o3d.utility.Vector3dVector(np.asarray(small_prev_cam.colors)[:8])
        small_prev_cam.colors = o3d.utility.Vector3dVector([camera_color for _ in range(8)])
        vis.add_geometry(small_prev_cam)
    
    vis.remove_geometry(vis_cur_cam)
    vis_pre_cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=224, view_height_px=224,
                                                                   intrinsic=intri, extrinsic=est_w2c, scale=1.0)
    vis_pre_cam.lines = o3d.utility.Vector2iVector(np.asarray(vis_pre_cam.lines)[:8])
    vis_pre_cam.colors = o3d.utility.Vector3dVector(np.asarray(vis_pre_cam.colors)[:8])
    vis_pre_cam.colors = o3d.utility.Vector3dVector([pre_camera_color for _ in range(8)])
    vis.add_geometry(vis_pre_cam)
    
    
    last_w2c = est_w2c.copy()
    
    
   




