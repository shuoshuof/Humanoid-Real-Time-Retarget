# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/21 16:52
@Auth ： shuoshuof
@File ：base_visualizer.py
@Project ：Humanoid-Real-Time-Retarget
"""

from typing import List,Union,Tuple
from abc import ABC,abstractmethod
import math
import copy
import time

from vedo import *

from vedo_visualizer import BaseRobot

settings.default_backend = "vtk"
settings.immediate_rendering = False

class BaseVedoVisualizer(ABC):
    def __init__(self,num_subplots,**kwargs):
        self.num_subplots = num_subplots
        self._init_plotter()
        self.counter = 0
        self.timer_id = None

        self._add_objects()

        self._init_widgets()
        self._init_callbacks()

    def _init_widgets(self):
        self.pause_button = self.plotter.at(self.num_subplots-1).add_button(
            self._stop_button_func,
            states=["\u23F5 Play", "\u23F8 Pause"],
            font="Kanopus",
            size=32,
        )

    def _init_callbacks(self):
        self.plotter.add_callback('timer', self.loop, enable_picking=False)

    def _init_plotter(self):
        cols = int(math.sqrt(self.num_subplots))
        rows = int(math.ceil(self.num_subplots / cols))
        self.plotter = Plotter(shape=(cols, rows), sharecam=False)

    @abstractmethod
    def _add_objects(self):
        pass

    def _stop_button_func(self, obj, ename):
        if self.timer_id is not None:
            self.plotter.timer_callback("destroy",self.timer_id)
        if "Play" in self.pause_button.status():
            self.timer_id = self.plotter.timer_callback("create",dt=5)
        self.pause_button.switch()

    @abstractmethod
    def loop(self,event):
        self.counter += 1

    def show(self):
        # self.plotter.timer_callback("start")
        self.plotter.interactive()

class RobotVisualizer(BaseVedoVisualizer):
    def __init__(self, num_subplots, robots:List[Union[BaseRobot]], data:List, **kwargs):
        self.robots = robots
        self.num_robot = len(robots)
        super().__init__(num_subplots, **kwargs)
        self.data = data
        assert self.num_robot == len(self.data)==self.num_subplots

    def _add_objects(self):
        for i in range(self.num_subplots):
            self.plotter.at(i).add(*self.robots[i].geoms)
            # self.plotter.at(i).show(axes=0, resetcam=False)
            self.plotter.at(i).render()
    def update_plt(self):
        for i in range(self.num_subplots):
            self.plotter.at(i).add(*self.robots[i].geoms)
            self.plotter.at(i).render()
            print(self.robots[i].geoms)
    @abstractmethod
    def update_robots(self):
        raise NotImplementedError
    def loop(self, event):
        start = time.time()
        self.counter +=1
        self.update_robots()
        self.update_plt()
        # print(f'fps: {round((1/(time.time()-start)),2)}')

class SkeletonRobotVisualizer(RobotVisualizer):
    def __init__(self, num_subplots, robots:List[Union[BaseRobot]], data:List, **kwargs):
        super().__init__(num_subplots, robots, data, **kwargs)

    def update_plt(self):
        for i in range(self.num_subplots):
            self.plotter.at(i).remove('Spheres')
            self.plotter.at(i).remove('Lines')
            self.plotter.at(i).remove('Arrows')
            self.plotter.at(i).add(*self.robots[i].geoms)
            self.plotter.at(i).render()

    def update_robots(self):
        for i in range(self.num_subplots):
            if self.counter>=len(self.data[i]):
                continue
            self.robots[i].forward(self.data[i][self.counter])

if __name__ == '__main__':
    pass