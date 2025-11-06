from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np

def interpolate_poses(pose1, pose2, alpha):
    """
    Interpolate between two poses.
    :param pose1: Initial pose (4x4 matrix)
    :param pose2: Final pose (4x4 matrix)
    :param alpha: Interpolation factor (0 <= alpha <= 1)
    :return: Interpolated pose (4x4 matrix)
    """
    # Interpolate translation
    translation1 = pose1[:3, 3]
    translation2 = pose2[:3, 3]
    translation_interp = (1 - alpha) * translation1 + alpha * translation2

    # Interpolate rotation using slerp
    rotation1 = R.from_matrix(pose1[:3, :3])
    rotation2 = R.from_matrix(pose2[:3, :3])
    slerp = Slerp([0,1],R.concatenate([rotation1, rotation2]))
    rotation_interp = slerp([alpha]).as_matrix()[0]
    # rotation_interp = R.slerp([0, 1], [rotation1, rotation2], [alpha]).as_matrix()[0]

    # Construct the interpolated pose
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = rotation_interp
    interpolated_pose[:3, 3] = translation_interp

    return interpolated_pose