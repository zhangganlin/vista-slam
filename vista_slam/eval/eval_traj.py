import numpy as np
import os

def align_traj(traj_est_all,traj_ref_all):
    '''
        traj_est: estimated trajectory list[np.array[4,4]]
        traj_ref: reference trajectory list[np.array[4,4]]
    '''
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    timestamps = []
    traj_ref = []
    traj_est = []
    for i in range(len(traj_ref_all)):
        val = traj_ref_all[i].sum()
        if np.isnan(val) or np.isinf(val):
            print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
            continue
        traj_est.append(traj_est_all[i])
        traj_ref.append(traj_ref_all[i])
        timestamps.append(float(i))

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
    return r_a, t_a, s, traj_est, traj_ref    


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name):
    import os
    from evo.core import metrics
    from evo.tools import plot
    import matplotlib.pyplot as plt
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)
    print("Calculating APE ...")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    print("Plotting ...")

    plot_collection = plot.PlotCollection("kf factor graph")
    # metric values
    fig_1 = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig_1, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(
        ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
        max_map=ape_statistics["max"], title="APE mapped onto trajectory"
    )
    plot_collection.add_figure("2d", fig_1)
    plot_collection.export(f"{plot_parent_dir}/{plot_name}.png", False)

    return ape_statistics


def full_traj_eval(traj_est, traj_ref, plot_parent_dir, plot_name):
    '''
    traj_est: estimated trajectory list[np.array[4,4]]
    traj_ref: reference trajectory list[np.array[4,4]]
    '''

    r_a, t_a, s, traj_est, traj_ref = align_traj(traj_est, traj_ref)    

    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name)
    
    return traj_est, traj_ref, r_a, t_a, s, ape_statistics