import pybullet as bullet
import numpy as np

import os
from rock_walk.resources.utils import *

class Cone:
    def __init__(self, client):
        self.clientID = client
        f_name = os.path.join(os.path.dirname(__file__),
                              'models/large_cone.urdf')

        self.coneID = bullet.loadURDF(fileName=f_name,
                                    basePosition=[0, 0, 0],
                                    physicsClientId=client)

    def get_ids(self):
        return self.coneID, self.clientID

    def apply_action(self, action):
        force = [0, action[0], 0]
        position = [0,-0.2,1.15]
        bullet.applyExternalForce(self.coneID, -1, force, position, bullet.LINK_FRAME)

    def get_observation(self):

        lin_pos_base_world, quat_base_world = bullet.getBasePositionAndOrientation(self.coneID, self.clientID)
        rot_base_world = bullet.getMatrixFromQuaternion(quat_base_world, self.clientID)
        rot_body_world = transform_to_body_frame(rot_base_world)

        psi, theta, phi = compute_body_euler(rot_body_world)

        lin_vel_base_world, ang_vel_base_world = bullet.getBaseVelocity(self.coneID, self.clientID)

        psi_dot, theta_dot, phi_dot = compute_body_velocity(rot_body_world, ang_vel_base_world)

        return [theta]
