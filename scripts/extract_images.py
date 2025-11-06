import numpy as np
import cv2

input_folder = f"bf_apt0"

output_folder_i = f"apt0_i"
output_folder_j = f"apt0_j"
output_folder_loop = f"apt0_loop"

images = (np.load(f"{input_folder}/images.npy")*255).astype(np.uint8)
view_graph_npz = np.load(f"{input_folder}/view_graph.npz",allow_pickle=True)
view_graph = view_graph_npz['view_graph'].item()
view_names = view_graph_npz['view_names'].tolist()
loop_min_dist = view_graph_npz['loop_min_dist'].item()

last_loop_img = None
for v in range(len(images)):
    view_j = images[v]
    if v ==0:
        view_i = images[v]
    else:
        view_i = images[v-1]
    loop_views = view_graph[v]
    view_loop = None
    for loop in loop_views:
        if abs(loop - v) > loop_min_dist and v > loop:
            view_loop = images[loop]
            break 
    view_i_img = view_i[...,[2,1,0]]
    view_j_img = view_j[...,[2,1,0]]
    if view_loop is None:
        view_loop_img = last_loop_img
        if view_loop_img is None:
            view_loop_img = np.zeros_like(view_i_img)
        else:
            view_loop_img = last_loop_img*0.2
    else:
        view_loop_img = view_loop[...,[2,1,0]]
        last_loop_img = view_loop_img
    # cv2.imwrite(f"{output_folder_i}/{v:04d}.png",view_i_img)
    # cv2.imwrite(f"{output_folder_j}/{v:04d}.png",view_j_img)          
    cv2.imwrite(f"{output_folder_loop}/{v:04d}.png",view_loop_img)          
              
