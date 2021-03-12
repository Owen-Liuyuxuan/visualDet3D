from numba import jit
import math
import numpy as np
from .bbox3d import project_3d
from .bbox2d import iou_2d
from visualDet3D.utils.utils import convertAlpha2Rot, convertRot2Alpha
def post_opt(bbox_2d, bbox3d_state_3d, P2, cx, cy):
    """
        run hill climbing algorithm
    """
    p2 = np.eye(4)
    p2[0:3] = P2.copy()
    p2_inv = np.linalg.inv(p2)
    box_2d = bbox_2d.detach().cpu().numpy()
    state = bbox3d_state_3d.detach().cpu().numpy()
    x, y, z, w, h, l, alpha = state[0], state[1], state[2],state[3],state[4],state[5], state[6]
    theta = convertAlpha2Rot(np.array([alpha]), cx, P2)[0]
    theta, ratio, w, h, l = post_optimization(p2, p2_inv, box_2d, cx, cy, z,
                                       w, h, l, theta, step_r_init=0.4, r_lim=0.01)
    z = z*ratio
    
    alpha = convertRot2Alpha(np.array([theta]), cx, P2)[0]
    return bbox3d_state_3d.new([cx,cy,z, w,h,l,alpha])

@jit(nopython=True, cache=True)
def post_optimization(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_r_init=0.3, r_lim=0.01):
    ratios = [1.0] # [0.95, 0.97, 1.0, 1.03, 1.05] #[0.95, 0.97, 1.0, 1.03, 1.05]
    ws = [w3d] # [w3d, w3d - 0.05, w3d + 0.05]
    hs = [h3d] # [h3d, h3d - 0.04, h3d + 0.04]
    ls = [l3d] # [l3d, l3d - 0.2 , l3d + 0.2 ]
    best_iou = -1e9
    best_theta = ry3d
    best_ratio = 1.0
    best_w = w3d
    best_h = h3d
    best_l = l3d
    for i in range(len(ratios)):
        z = z2d * ratios[i]
        for w in ws:
            for h in hs:
                for l in ls:
                    theta, iou = hill_climb(p2, p2_inv, box_2d, x2d, y2d, z, w, h, l, ry3d, step_r_init=step_r_init, r_lim=r_lim)
                    if iou > best_iou:
                        best_iou = iou
                        best_theta = theta
                        best_ratio = ratios[i]
                        best_w = w
                        best_h = h
                        best_l = l

    return best_theta,  best_ratio, best_w, best_h, best_l

@jit(nopython=True)
def hill_climb(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_r_init, r_lim=0.0,min_ol_dif=0.0):
    step_r = step_r_init
    ol_best = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)
    # attempt to fit z/rot more properly
    while (step_r > r_lim):

        if step_r > r_lim:
            ol_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d - step_r)
            ol_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d + step_r)
            
            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg:
                ry3d += step_r
                ol_best = ol_pos
            elif (ol_neg - ol_best) > min_ol_dif:
                ry3d -= step_r
                ol_best = ol_neg
            else:
                step_r = step_r * 0.5

    while ry3d > 3.14: ry3d -= 3.14 * 2
    while ry3d < (-3.14): ry3d += np.pi * 2

    return ry3d, ol_best


@jit(nopython=True)
def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    x = box_2d[0]
    y = box_2d[1]
    x2 = box_2d[2]
    y2 = box_2d[3]

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    #top_3d = p2_inv.dot(np.array([cx * z, (cy-h3d/2) * z, z, 1]))
    # put back on ground first
    #cy3d += h3d/2
    fy = p2[1, 1]

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY)
    #verts3d = np.array([[0, 0], [0, 0]])
    x_new = min(verts3d[:, 0])
    x_new = max(0, x_new)
    y_new = min(verts3d[:, 1])
    y_new = max(0, y_new)
    #y_new = top_3d[1]
    x2_new = max(verts3d[:, 0])
    x2_new = min(x2_new, 1280)
    y2_new = max(verts3d[:, 1])
    y2_new = min(y2_new, 288)

    b1 = np.array([x, y, x2, y2]).reshape((1, 4))
    b2 = np.array([x_new, y_new, x2_new, y2_new]).reshape((1,4))
    #ol = -(np.abs(x - x_new) + 0.5 * np.abs(y - y_new) + np.abs(x2 - x2_new) + 0.5*np.abs(y2 - y2_new)  + 0.5*np.abs(bot - bot_new) + 0.5*np.abs(top - top_new)) 
    ol = iou_2d(b1, b2)[0]
    
    #ol = -(np.abs(x - x_new) + np.abs(x2 - x2_new))

    return ol


if __name__ == '__main__':
    

    p2 = np.array([[ 5.02790613e+02,  0.00000000e+00,  4.29568996e+02,  3.25392427e+01],
 [ 0.00000000e+00,  5.02790613e+02,  5.72491378e+01, -5.99834524e-01],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  4.98101600e-03],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    p2_inv = np.array([[ 0.0019889,   0.,         -0.85436956, -0.06046166],
 [ 0.,          0.0019889,  -0.11386278,  0.00176016],
 [ 0.,          0.,          1.,         -0.00498102],
 [ 0.,          0.,          0. ,         1.        ]])
    box_2d = np.array([490.3174,   64.63407, 568.4109,  105.2571 ])
    x2d, y2d, z2d, w3d, h3d, l3d, ry3d =  528.2042846679688,82.82894134521484, 20.556593, 1.5336921, 1.4364641, 3.3523552, 1.6921594


    best_theta,  best_ratio,_,_,_ = post_optimization(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d,step_r_init=0.4, r_lim=0)
    print(best_theta)
