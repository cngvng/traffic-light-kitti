import os

import matplotlib.pyplot as plt
import open3d

from utils import *

import sys

sys.path.append("/workspaces/PheNet-Traffic_light")

from test_code.utils import *




def render_lidar_on_image_focus_label(pts_velo, img, calib, labels, label_color):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i, label in enumerate(labels):
        # Filter lidar points within a certain radius of the label location
        radius = 5 # define the radius around the label location
        # label = map_box_to_image(label, P_rect2cam2)
        label = map_box_to_image(label, proj_velo2cam2)
        dist = np.sqrt((pts_2d[0,:] - label[0])**2 + (pts_2d[1,:] - label[1])**2)
        inds = np.where(dist < radius)[0]

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d[:, inds]

        # Retrieve depth from lidar
        imgfov_pc_velo = pts_velo[inds, :]
        imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

        for j in range(imgfov_pc_pixel.shape[1]):
            depth = imgfov_pc_cam2[2, j]
            color = label_color if dist[inds[j]] < 1 else cmap[int(640.0 / depth), :]
            cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, j])),
                             int(np.round(imgfov_pc_pixel[1, j]))),
                       2, color=tuple(color), thickness=-1)

    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img


if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(os.path.join('/workspaces/PheNet-Traffic_light/test_code/data/000084_image.png')), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('/workspaces/PheNet-Traffic_light/test_code/data/000084_calib.txt')

    # Load labels
    # labels = load_label('/home/cngvng/Phenikaa-X/cvml_project/projections/lidar_camera_projection/data/000114_label.txt')
    labels = load_label('/workspaces/PheNet-Traffic_light/test_code/data/000084_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan('/workspaces/PheNet-Traffic_light/test_code/data/000084.bin')[:, :3]

    
    render_lidar_on_image_focus_label(pc_velo, rgb, calib,  labels, (255, 255, 0))
