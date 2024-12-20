# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from ..core import logger
from .plt_plotter import Matplotlib3DPlotter, Matplotlib3DPlotterAx
from .skeleton_plotter_tasks import Draw3DSkeletonMotion, Draw3DSkeletonState, Draw3DMotionDICT


def plot_skeleton_state(skeleton_state, task_name=""):
    """
    Visualize a skeleton state

    :param skeleton_state:
    :param task_name:
    :type skeleton_state: SkeletonState
    :type task_name: string, optional
    """
    logger.info("plotting {}".format(task_name))
    task = Draw3DSkeletonState(task_name=task_name, skeleton_state=skeleton_state)
    plotter = Matplotlib3DPlotter(task)
    plotter.show()

def plot_skeleton_state_M(skeleton_states, task_name=""):
    """
    Visualize a skeleton state

    :param skeleton_state:
    :param task_name:
    :type skeleton_state: SkeletonState
    :type task_name: string, optional
    """
    logger.info("plotting {}".format(task_name))
    tasks = []
    for skeleton_state in skeleton_states:
        tasks.append(Draw3DSkeletonState(task_name=task_name, skeleton_state=skeleton_state))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 5), subplot_kw={'projection': '3d'})

    plotters = []
    for i, task in enumerate(tasks):
        plotters.append(Matplotlib3DPlotterAx(task=task, fig=fig, ax=axs[i]))
    [plotters[i].show() for i in range(len(tasks))]


def plot_skeleton_states(skeleton_state, skip_n=1, task_name=""):
    """
    Visualize a sequence of skeleton state. The dimension of the skeleton state must be 1

    :param skeleton_state:
    :param task_name:
    :type skeleton_state: SkeletonState
    :type task_name: string, optional
    """
    logger.info("plotting {} motion".format(task_name))
    assert len(skeleton_state.shape) == 1, "the state must have only one dimension"
    task = Draw3DSkeletonState(task_name=task_name, skeleton_state=skeleton_state[0])
    plotter = Matplotlib3DPlotter(task)
    for frame_id in range(skeleton_state.shape[0]):
        if frame_id % skip_n != 0:
            continue
        task.update(skeleton_state[frame_id])
        plotter.update()
    plotter.show()


def plot_skeleton_motion(skeleton_motion, skip_n=1, task_name=""):
    """
    Visualize a skeleton motion along its first dimension.

    :param skeleton_motion:
    :param task_name:
    :type skeleton_motion: SkeletonMotion
    :type task_name: string, optional
    """
    logger.info("plotting {} motion".format(task_name))
    task = Draw3DSkeletonMotion(
        task_name=task_name, skeleton_motion=skeleton_motion, frame_index=0
    )
    plotter = Matplotlib3DPlotter(task)
    for frame_id in range(len(skeleton_motion)):
        if frame_id % skip_n != 0:
            continue
        task.update(frame_id)
        plotter.update()
    plotter.show()

def plot_skeleton_motion_M(skeleton_motions, skip_n=1, task_name="", ax=None):
    """
    Visualize a skeleton motion along its first dimension.

    :param skeleton_motion:
    :param task_name:
    :param ax: Matplotlib axis object to plot on
    :type skeleton_motion: SkeletonMotion
    :type task_name: string, optional
    :type ax: matplotlib.axes.Axes, optional
    """
    logger.info("plotting {} motion".format(task_name))
    if type(skeleton_motions) != list:
        skeleton_motions = [skeleton_motions]

    number = len(skeleton_motions)

    tasks = []
    for skeleton_motion in skeleton_motions:
        tasks.append(Draw3DSkeletonMotion(
            task_name=task_name, skeleton_motion=skeleton_motion, frame_index=0
        ))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 5), subplot_kw={'projection': '3d'})

    plotters = []
    for i, task in enumerate(tasks):
        plotters.append(Matplotlib3DPlotterAx(task=task, fig=fig, ax=axs[i] if type(axs) == list else axs, boundary=True))

    for frame_id in range(len(skeleton_motions[0])):
        if frame_id % skip_n != 0:
            continue
        [tasks[i].update(frame_id) for i in range(number)]
        [plotters[i].update() for i in range(number)]
    [plotters[i].show() for i in range(number)]
    # plt.show()


# SkeletonState SkeletonMotion 
def plot_skeleton_H(sks, task_name=""):
    logger.info("plotting {} motion".format(task_name))
    number = len(sks)

    tasks = []
    types = []
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState, MotionDICT
    for sk in sks:
        if type(sk) == SkeletonState:
            tasks.append(Draw3DSkeletonState(task_name=task_name, skeleton_state=sk))
            types.append("state")
        elif type(sk) == SkeletonMotion:
            tasks.append(Draw3DSkeletonMotion(task_name=task_name, skeleton_motion=sk, frame_index=0))
            types.append("motion")
        elif type(sk) == MotionDICT:
            tasks.append(Draw3DMotionDICT(task_name=task_name, skeleton_motion=sk, frame_index=0))
            types.append("motion")
        else:
            raise ValueError("Invalid type of skeleton")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 5), subplot_kw={'projection': '3d'})

    plotters = []
    for i, task in enumerate(tasks):
        plotters.append(Matplotlib3DPlotterAx(task=task, fig=fig, ax=axs[i] if number>1 else axs))

    if 'motion' in types:
        for frame_id in range(len(sks[types.index("motion")])):
            [tasks[i].update(frame_id)  for i in range(number) if types[i] == 'motion']
            [plotters[i].update() for i in range(number)]
    else:
        [plotters[i].show() for i in range(number)]




def plot_skeleton_motion_interactive_base(skeleton_motion, task_name=""):
    class PlotParams:
        def __init__(self, total_num_frames):
            self.current_frame = 0
            self.playing = False
            self.looping = False
            self.confirmed = False
            self.playback_speed = 4
            self.total_num_frames = total_num_frames

        def sync(self, other):
            self.current_frame = other.current_frame
            self.playing = other.playing
            self.looping = other.current_frame
            self.confirmed = other.confirmed
            self.playback_speed = other.playback_speed
            self.total_num_frames = other.total_num_frames

    task = Draw3DSkeletonMotion(
        task_name=task_name, skeleton_motion=skeleton_motion, frame_index=0
    )
    plotter = Matplotlib3DPlotter(task)

    plot_params = PlotParams(total_num_frames=len(skeleton_motion))
    print("Entered interactive plot - press 'n' to quit, 'h' for a list of commands")

    def press(event):
        if event.key == "x":
            plot_params.playing = not plot_params.playing
        elif event.key == "z":
            plot_params.current_frame = plot_params.current_frame - 1
        elif event.key == "c":
            plot_params.current_frame = plot_params.current_frame + 1
        elif event.key == "a":
            plot_params.current_frame = plot_params.current_frame - 20
        elif event.key == "d":
            plot_params.current_frame = plot_params.current_frame + 20
        elif event.key == "w":
            plot_params.looping = not plot_params.looping
            print("Looping: {}".format(plot_params.looping))
        elif event.key == "v":
            plot_params.playback_speed *= 2
            print("playback speed: {}".format(plot_params.playback_speed))
        elif event.key == "b":
            if plot_params.playback_speed != 1:
                plot_params.playback_speed //= 2
            print("playback speed: {}".format(plot_params.playback_speed))
        elif event.key == "n":
            plot_params.confirmed = True
        elif event.key == "h":
            rows, columns = os.popen("stty size", "r").read().split()
            columns = int(columns)
            print("=" * columns)
            print("x: play/pause")
            print("z: previous frame")
            print("c: next frame")
            print("a: jump 10 frames back")
            print("d: jump 10 frames forward")
            print("w: looping/non-looping")
            print("v: double speed (this can be applied multiple times)")
            print("b: half speed (this can be applied multiple times)")
            print("n: quit")
            print("h: help")
            print("=" * columns)

        print(
            'current frame index: {}/{} (press "n" to quit)'.format(
                plot_params.current_frame, plot_params.total_num_frames - 1
            )
        )

    plotter.fig.canvas.mpl_connect("key_press_event", press)
    while True:
        reset_trail = False
        if plot_params.confirmed:
            break
        if plot_params.playing:
            plot_params.current_frame += plot_params.playback_speed
        if plot_params.current_frame >= plot_params.total_num_frames:
            if plot_params.looping:
                plot_params.current_frame %= plot_params.total_num_frames
                reset_trail = True
            else:
                plot_params.current_frame = plot_params.total_num_frames - 1
        if plot_params.current_frame < 0:
            if plot_params.looping:
                plot_params.current_frame %= plot_params.total_num_frames
                reset_trail = True
            else:
                plot_params.current_frame = 0
        yield plot_params
        task.update(plot_params.current_frame, reset_trail)
        plotter.update()


def plot_skeleton_motion_interactive(skeleton_motion, task_name=""):
    """
    Visualize a skeleton motion along its first dimension interactively.

    :param skeleton_motion:
    :param task_name:
    :type skeleton_motion: SkeletonMotion
    :type task_name: string, optional
    """
    for _ in plot_skeleton_motion_interactive_base(skeleton_motion, task_name):
        pass


def plot_skeleton_motion_interactive_multiple(*callables, sync=True):
    for _ in zip(*callables):
        if sync:
            for p1, p2 in zip(_[:-1], _[1:]):
                p2.sync(p1)


# def plot_skeleton_motion_interactive_multiple_same(skeleton_motions, task_name=""):

