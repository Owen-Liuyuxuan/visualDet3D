from numba import jit
import numpy as np

@jit
def convertAlpha2Rot(alpha, z3d, x3d):
    ry3d = alpha + np.arctan2(-z3d, x3d) + 0.5 * np.pi
    ry3d[np.where(ry3d > np.pi)] -= 2 * np.pi
    ry3d[np.where(ry3d <= -np.pi)] += 2 * np.pi
    return ry3d

@jit
def convertRot2Alpha(ry3d, z3d, x3d):

    alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * np.pi
    alpha[alpha > np.pi] -= 2 * np.pi
    alpha[alpha <= -np.pi] += 2 * np.pi
    return alpha

@jit
def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis

    return:
        verts3d:[8, 2], image coordinate of each corner points, 
        corners_3d: [8,3], camera coordinate of each corner points in 3d

        top_index:    [0, 1, 4, 5]
        bottom_index: [2, 3, 6, 7]
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+np.cos(ry3d), 0.0, +np.sin(ry3d)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(ry3d), 0.0, +np.cos(ry3d)]])
    
    # 3D bounding box corners
    x_corners = np.array([0.0, l3d, l3d, l3d, l3d,   0.0,   0.0,   0.0])
    y_corners = np.array([0.0, 0.0,   h3d, h3d,   0.0,   0.0, h3d, h3d])
    z_corners = np.array([0.0, 0.0,     0.0, w3d, w3d, w3d, w3d,   0.0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # # bounding box in object co-ordinate
    corners_3d = np.zeros((3, 8))
    for i in range(8):
        corners_3d[0, i] = x_corners[i]
        corners_3d[1, i] = y_corners[i]
        corners_3d[2, i] = z_corners[i]
    # corners_3d = np.stack([x_corners, y_corners, z_corners], axis=0)
    # corners_3d = np.array([0.0,0.0,0.0])
    # rotate
    corners_3d = np.dot(R, corners_3d)#R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.ones((4, 8))
    for i in range(3):
        corners_3D_1[i] = corners_3d[i]
    # corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    #bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = np.transpose(corners_2D[:2])

    #verts3d = np.array([[0, 0], [0, 0]]) 
    return verts3d, corners_3d


if __name__ == '__main__':
    alpha = np.array([0.5])
    z3d = 20.0
    x3d = 2.0
    theta = convertAlpha2Rot(alpha, z3d, x3d)
    alpha1 = convertRot2Alpha(theta, z3d, x3d)
    print(alpha, alpha1, theta)
    