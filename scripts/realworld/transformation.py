

import numpy as np

from pyquaternion import Quaternion

def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def transform_position(trans, x, y, z):
    src = np.array([[x], [y], [z], [1]])
    return np.dot(trans, src)

# def transform_pose(pose, ptrans, qtrans):
#     l_pos = np.array([[self.lx], [self.ly], [self.lz], [1]])
#     g_pos = np.dot(l2g_transform, l_pos)
#     g_quat = l2g_quat_transform * l_quat
#     g_rpy = quaternion_to_euler(g_quat)
#     self.gx, self.gy, self.gz, self.grz = g_pos[0][0], g_pos[1][0], g_pos[2][0], g_rpy[2]
#     l_pos = np.array([[self.lx], [self.ly], [self.lz], [1]])
#     g_pos = np.dot(l2g_transform, l_pos)
#     self.tgt_q = qtransform * src_q
#     return trans_pose

def get_transform_between_poses(src_pose, tgt_pose):

    tx = src_pose.position.x - tgt_pose.position.x
    ty = src_pose.position.y - tgt_pose.position.y
    tz = src_pose.position.z - tgt_pose.position.z
    t = np.array([[tx], [ty], [tz]])
    
    q_src = Quaternion(w=src_pose.orientation.w, x=src_pose.orientation.x, y=src_pose.orientation.y, z=src_pose.orientation.z)
    q_tgt = Quaternion(w=tgt_pose.orientation.w, x=tgt_pose.orientation.x, y=tgt_pose.orientation.y, z=tgt_pose.orientation.z)
    q_st = q_src * q_tgt.inverse
    R = quaternion_to_rotation_matrix(q_st)

    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def transform_from_gps_and_imu(gps, imu): # NavSatFix and Imu Messages # Origin (First T)
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    scale = np.cos(gps.latitude * np.pi / 180.)
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * gps.longitude * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + gps.latitude) * np.pi / 360.))
    tz = gps.altitude
    t = np.array([[tx], [ty], [tz]])

    # Use the Euler angles to get the rotation matrix
    # Rx = rotx(roll)
    # Ry = roty(pitch)
    # Rz = rotz(yaw)
    # R = Rz.dot(Ry.dot(Rx))

    # Quaternion to Matrix
    R = quaternion_to_rotation_matrix(Quaternion(w=imu.orientation.w, x=imu.orientation.x, y=imu.orientation.y, z=imu.orientation.z))

    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calibration(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass


        # print(data['Tr_imu_to_velo'])
        # data['P2'] = np.reshape(data['P2'], [3,4])
        # # Rigid transform from Velodyne coord to reference camera coord
        data['Tr_velo_to_cam'] = np.reshape(data['Tr_velo_to_cam'], [3,4])
        data['Tr_imu_to_velo'] = np.reshape(data['Tr_imu_to_velo'], [3,4])
        # print(data['Tr_imu_to_velo'])
        data['R0_rect'] = np.reshape(data['R0_rect'], [3,3])
        lidar2cam = np.eye(4, 4)
        imu2lidar = np.eye(4, 4)
        cam2center = np.eye(4, 4)
        lidar2cam[0:3, 0:4] = data['Tr_velo_to_cam']
        imu2lidar[0:3, 0:4] = data['Tr_imu_to_velo']
        # print(imu2lidar)
        cam2center[0:3, 0:3] = data['R0_rect']
        imu2cam = np.dot(cam2center, np.dot(lidar2cam, imu2lidar))
        quat_lidar2cam = Quaternion(matrix=lidar2cam[0:3, 0:3])
        quat_imu2lidar = Quaternion(matrix=imu2lidar[0:3, 0:3])
        quat_cam2center = Quaternion(matrix=cam2center)
        quat_imu2cam = quat_cam2center * quat_lidar2cam * quat_imu2lidar
        cam2imu = np.linalg.inv(imu2cam)
        quat_cam2imu = quat_imu2cam.inverse
        cam2lidar = np.linalg.inv(lidar2cam)
        quat_cam2lidar = quat_lidar2cam.inverse
        # transforms = {'Pvelo': data['P2'], 'velo2cam': velo2cam, 'qvelo2cam': qvelo2cam, 'imu2cam': imu2cam, 'qimu2cam': qimu2cam,
        #               'cam2imu': cam2imu, 'qcam2imu': qcam2imu, 'cam2velo': cam2velo, 'qcam2velo': qcam2velo}
        
        lidar2imu = imu2lidar.T # np.linalg.inv(imu2lidar)
        # print(np.dot(imu2lidar, lidar2imu))
        quat_lidar2imu = quat_imu2lidar.inverse
        transforms = {'imu2lidar': imu2lidar, 'quat_imu2lidar': quat_imu2lidar, 'lidar2imu': lidar2imu, 'quat_lidar2imu': quat_lidar2imu,
                      'lidar2cam': lidar2cam, 'quat_lidar2cam': quat_lidar2cam, 'cam2lidar': cam2lidar, 'quat_cam2lidar': quat_cam2lidar,
                      'imu2cam': imu2cam, 'quat_imu2cam': quat_imu2cam, 'cam2imu': cam2imu, 'quat_cam2imu': quat_cam2imu,
                      'cam2center': cam2center, 'quat_cam2center': quat_cam2center}
    return transforms



def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def quaternion_to_euler(q):
    t0 = +2.0 * (q[0] * q[1] + q[2] * q[3])
    t1 = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (q[0] * q[2] - q[3] * q[1])
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (q[0] * q[3] + q[1] * q[2])
    t4 = +1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(t3, t4)

    # if abs(yaw) > 1.0:
    #     if pitch > 0:
    #         pitch = np.pi - pitch
    #     else:
    #         pitch = -np.pi - pitch

    # if abs(rpy[2]) > 1.0:
    #     if rpy[1] > 0:
    #         rot_y = np.pi - rpy[1]
    #     else:
    #         rot_y = -np.pi - rpy[1]
    # else:
    #     rot_y = rpy[1]

    return [roll, pitch, yaw]

def quaternion_to_rotation_matrix(q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

# def cart2hom(self, pts_3d):
#     ''' Input: nx3 points in Cartesian
#         Oupput: nx4 points in Homogeneous by pending 1
#     '''
#     n = pts_3d.shape[0]
#     pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
#     return pts_3d_hom

# # =========================== 
# # ------- 3d to 2d ---------- 
# # =========================== 
# def project_rect_to_image(self, pts_3d_rect, projection_mtx):
#     ''' Input: nx3 points in rect camera coord.
#         Output: nx2 points in image2 coord.
#     '''
#     pts_3d_rect = self.cart2hom(pts_3d_rect)
#     pts_2d = np.dot(pts_3d_rect, np.transpose(projection_mtx)) # nx3
#     pts_2d[:,0] /= pts_2d[:,2]
#     pts_2d[:,1] /= pts_2d[:,2]
#     return pts_2d[:,0:2]

# def project_velo_to_image(self, pts_3d_velo, projection_mtx):
#     ''' Input: nx3 points in velodyne coord.
#         Output: nx2 points in image2 coord.
#     '''
#     pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
#     return self.project_rect_to_image(pts_3d_rect, projection_mtx)

# # =========================== 
# # ------- 2d to 3d ---------- 
# # =========================== 
# def project_image_to_rect(self, uv_depth):
#     ''' Input: nx3 first two channels are uv, 3rd channel
#                 is depth in rect camera coord.
#         Output: nx3 points in rect camera coord.
#     '''
#     n = uv_depth.shape[0]
#     x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
#     y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
#     pts_3d_rect = np.zeros((n,3))
#     pts_3d_rect[:,0] = x
#     pts_3d_rect[:,1] = y
#     pts_3d_rect[:,2] = uv_depth[:,2]
#     return pts_3d_rect

# def project_image_to_velo(self, uv_depth):
#     pts_3d_rect = self.project_image_to_rect(uv_depth)
#     return self.project_rect_to_velo(pts_3d_rect)

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        # modified from https://github.com/kuixu/kitti_object_vis/blob/master/kitti_util.py, MIT license
    '''
    def __init__(self, calib_filepath, from_video=False):
        # if from_video:
        #     calibs = self.read_calib_from_video(calib_filepath)
        # else:
        calibs = self.read_calib_file(calib_filepath)

        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'] 
        self.P = np.reshape(self.P, [3,4])

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.V2C_R = self.V2C[:3, :3]
        self.V2C_T = self.V2C[:, 3]
        self.C2V = inverse_rigid_trans(self.V2C)
        
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        self.I2V = calibs['Tr_imu_to_velo']  # 3 x 4
        self.I2V = np.reshape(self.I2V, [3,4])
        self.V2I = inverse_rigid_trans(self.I2V)

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]              
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def imu_to_rect(self, pts_imu):
        pts_velo = self.imu_to_velo(pts_imu)
        pts_ref = self.project_velo_to_ref(pts_velo)
        pts_rect = self.project_ref_to_rect(pts_ref)
        return pts_rect

    def imu_to_velo(self, pts_imu):
        pts_imu = self.cart2hom(pts_imu) # nx4
        return np.dot(pts_imu, np.transpose(self.I2V))

    def velo_to_imu(self, pts_velo):
        pts_velo = self.cart2hom(pts_velo) # nx4
        return np.dot(pts_velo, np.transpose(self.V2I))

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def rect_to_imu(self, pts_rect):
        pts_velo = self.project_rect_to_velo(pts_rect)
        pts_imu  = self.velo_to_imu(pts_velo)

        return pts_imu

    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)