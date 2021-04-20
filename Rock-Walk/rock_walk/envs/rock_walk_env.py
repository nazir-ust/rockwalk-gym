import gym
import math
import numpy as np
import pybullet as bullet
import matplotlib.pyplot as plt

from rock_walk.resources.cone import Cone
from rock_walk.resources.plane import Plane


class RockWalkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = gym.spaces.box.Box(
            low=np.array([-2.5], dtype=np.float32),
            high=np.array([0], dtype=np.float32))

        self.observation_space = gym.spaces.box.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([np.pi/2], dtype=np.float32))

        self.np_random, _ = gym.utils.seeding.np_random()

        self.clientID = bullet.connect(bullet.GUI)
        # self.clientID = bullet.connect(bullet.DIRECT)
        # Reduce length of episodes for RL algorithms
        bullet.setTimeStep(1/30.0, self.clientID)


        self.cone = None
        self.goal = [0.7]
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()



    def step(self, action):
        self.cone.apply_action(action)
        bullet.stepSimulation()
        cone_ob = self.cone.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = abs(cone_ob[0]-self.goal[0])

        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        if (cone_ob[0] >= 1.0):
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 1e-2:
            self.done = True
            reward = 50

        ob = np.array(cone_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()


    def reset(self):
        bullet.resetSimulation(self.clientID)
        bullet.setGravity(0, 0, -10)

        Plane(self.clientID)
        self.cone = Cone(self.clientID)

        # self.goal = [0.5]
        self.done = False
        cone_ob = self.cone.get_observation()

        self.prev_dist_to_goal = abs(cone_ob[0]-self.goal[0])

        return np.array(cone_ob + self.goal, dtype=np.float32)

    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
            # plt.show()

        # Base information
        cone_id, client_id = self.cone.get_ids()
        proj_matrix = bullet.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100)

        pos, ori = [list(l) for l in bullet.getBasePositionAndOrientation(cone_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(bullet.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = bullet.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = bullet.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)


    def close(self):
        bullet.disconnect(self.clientID)


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
