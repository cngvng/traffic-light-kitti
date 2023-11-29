import sys

sys.path.append("/workspaces/PheNet-Traffic_light")
from test_code.utils import *
import matplotlib.pyplot as plt

def project_point(K, X, Y, Z):
    fx, fy = K[0, 0], K[1, 1]  # Focal length
    cx, cy = K[0, 2], K[1, 2]  # Principal point
    s = K[0, 1]                # Skew factor
    
    u = (fx*X/Z + s*Y/Z + cx)  # Tính tọa độ điểm trên trục ngang (x)
    v = (fy*Y/Z + cy)         # Tính tọa độ điểm trên trục dọc (y)
    
    return u, v

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [3, 3]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px],
                     [0, fy, py],
                     [0, 0, 1.]])

def render_lidar_on_image_focus_label(img, calib, img_width, img_height, labels, label_color):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    K = intrinsic_from_fov(img_height, img_width)

    for i, label in enumerate(labels):
        X, Y, Z = label.in_camera_coordinate()
        u, v = project_point(K, X, Y, Z)
        # Tọa độ góc trên bên trái và góc dưới bên phải của hình chữ nhật
        cv2.circle(img, (u,v), 10, (0,0,255), 2)

    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img

if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread('/workspaces/PheNet-Traffic_light/test_code/data/000084_image.png'), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('/workspaces/PheNet-Traffic_light/test_code/data/000084_calib.txt')

    # Load labels
    # labels = load_label('/home/cngvng/Phenikaa-X/cvml_project/projections/lidar_camera_projection/data/000114_label.txt')
    labels = load_label('/workspaces/PheNet-Traffic_light/test_code/data/000084_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan('/workspaces/PheNet-Traffic_light/test_code/data/000084.bin')[:, :3]

    render_lidar_on_image_focus_label(rgb, calib, img_width, img_height, labels, (255, 255, 0))