import pybullet as bullet
import os


class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'models/plane.urdf')

        bullet.loadURDF(fileName=f_name,
                        basePosition=[0, 0, 0],
                        physicsClientId=client)
