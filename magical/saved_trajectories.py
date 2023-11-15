"""Tools for saving and loading trajectories without requiring the `imitation`
or `tensorflow` packages to be installed."""
import copy
import datetime
import gzip
import os
from pickle import Unpickler
from typing import List, NamedTuple, Optional
import cv2

import gym
import numpy as np
from magical.benchmarks import register_envs
from magical.benchmarks import (  # comment to stop yapf touching import
    DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS, update_magical_env_name)
from magical.render import Compound

class MAGICALTrajectory(NamedTuple):
    """Trajectory representation compatible with imitation's trajectory data
    class."""

    acts: np.ndarray
    obs: dict
    rews: np.ndarray
    infos: Optional[List[dict]]


class _TrajRewriteUnpickler(Unpickler):
    """Custom unpickler that replaces references to `Trajectory` class in
    `imitation` with custom trajectory class in this module."""
    def find_class(self, module, name):
        # print('find_class(%r, %r)' % (module, name))
        if (module, name) == ('imitation.util.rollout', 'Trajectory') \
          or (module, name) == ('milbench.baselines.saved_trajectories',
                                'MILBenchTrajectory'):
            return MAGICALTrajectory
        return super().find_class(module, name)


def load_demos(demo_paths, rewrite_traj_cls=True, verbose=False):
    """Use GzipFile & pickle to generate a sequence of demo dictionaries from a
    sequence of file paths."""
    n_demos = len(demo_paths)
    for d_num, d_path in enumerate(demo_paths, start=1):
        if verbose:
            print(f"Loading '{d_path}' ({d_num}/{n_demos})")
        with gzip.GzipFile(d_path, 'rb') as fp:
            if rewrite_traj_cls:
                unpickler = _TrajRewriteUnpickler(fp)
            else:
                unpickler = Unpickler(fp)
            this_dict = unpickler.load()
        yield this_dict


def splice_in_preproc_name(base_env_name, preproc_name):
    """Splice the name of a preprocessor into a magical benchmark name. e.g.
    you might start with "MoveToCorner-Demo-v0" and insert "LoResStack" to end
    up with "MoveToCorner-Demo-LoResStack-v0". Will do a sanity check to ensure
    that the preprocessor actually exists."""
    assert preproc_name in DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS, \
        f"no preprocessor named '{preproc_name}', options are " \
        f"{', '.join(DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS)}"
    return update_magical_env_name(base_env_name, preproc=preproc_name)


class _MockDemoEnv(gym.Wrapper):
    """Mock Gym environment that just returns an observation"""
    def __init__(self, orig_env, trajectory):
        super().__init__(orig_env)
        self._idx = 0
        self._traj = trajectory
        self._traj_length = len(self._traj.acts)

    def reset(self):
        self._idx = 0
        return self._traj.obs[self._idx]

    def step(self, action):
        rew = self._traj.rews[self._idx]
        info = self._traj.infos[self._idx] or {}
        info['_mock_demo_act'] = self._traj.acts[self._idx]
        self._idx += 1
        # ignore action, return next obs
        obs = self._traj.obs[self._idx]
        # it's okay if we run one over the end
        done = self._idx >= self._traj_length
        return obs, rew, done, info


def preprocess_demos_with_wrapper(trajectories,
                                  orig_env_name,
                                  preproc_name=None,
                                  wrapper=None):
    """Preprocess trajectories using one of the built-in environment
    preprocessing pipelines.

    Args:
        trajectories ([Trajectory]): list of trajectories to process.
        orig_env_name (str): name of original environment where trajectories
            were collected. This function will instantiate a temporary instance
            of that environment to get access to an observation space and other
            metadata.
        preproc_name (str or None): name of preprocessor to apply. Should be
            available in
            `magical.benchmarks.DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS`.
        wrapper (callable or None): wrapper constructor. Should take a
            constructed Gym environment and return a wrapped Gym-like
            environment. Either `preproc_name` or `wrapper` must be specified,
            but both cannot be specified at once.

    Returns:
        rv_trajectories ([Trajectory]): equivalent list of trajectories that
            have each been preprocessed with the given wrapper."""
    if preproc_name is not None:
        assert wrapper is None
        wrapper = DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS[preproc_name]
    else:
        assert wrapper is not None
    orig_env = gym.make(orig_env_name)
    wrapped_constructor = wrapper(_MockDemoEnv)
    rv_trajectories = []
    for traj in trajectories:
        mock_env = wrapped_constructor(orig_env=orig_env, trajectory=traj)
        obs = mock_env.reset()
        values = {
            'obs': [],
            'acts': [],
            'rews': [],
            'infos': [],
        }
        values['obs'].append(obs)
        done = False
        while not done:
            obs, rew, done, info = mock_env.step(None)
            acts = info['_mock_demo_act']
            del info['_mock_demo_act']
            values['obs'].append(obs)
            values['acts'].append(acts)
            values['rews'].append(rew)
            values['infos'].append(info)
        # turn obs, acts, and rews into numpy arrays
        stack_values = {
            k: np.stack(vs, axis=0)
            for k, vs in values.items() if k in ['obs', 'acts', 'rews']
        }
        # keep infos as a list (hard to get at elements otherwise)
        stack_values['infos'] = values.get('infos')
        # use type(traj) to preserve either MAGICAL trajectory type or custom
        # type
        new_traj = type(traj)(**stack_values)
        rv_trajectories.append(new_traj)
    return rv_trajectories


def rerender_from_geoms(demos, easy=True):
    """
    Re-render pixels of trajectories from their geoms. This is useful if to re-render if you have edited the rendering code and want to see the new rendering.
    If the trajectory was saved with geoms, then this will change th pixel values in ego and allocentric frames to the new rendering.
    Returns: list[MagicalTrajectory]
    """
    register_envs()
    for demo in demos:
        env_name = demo['env_name']
        traj = demo['trajectory']
        if 'geoms' not in traj.obs[0]:
            print("Trajectory does not have geoms, skipping")
            continue
        if easy:
            env  = gym.make(env_name, easy_visuals=True)
        else:
            env  = gym.make(env_name, easy_visuals=False)
        env.reset()
        for i in range(len(traj.obs)):
            env.renderer.geoms = copy.deepcopy(traj.obs[i]['geoms'])
            if not easy:
                for geom in env.renderer.geoms:
                    # blocks and robots
                    if type(geom) == Compound:
                        for subgeom in geom.gs:
                            subgeom.label = None
                    # goal regions
                    else:
                        geom.label = None
                    
            obs_dict = env.render('rgb_array')
            traj.obs[i]['ego'] = obs_dict['ego']
            traj.obs[i]['allo'] = obs_dict['allo']
    return demos



def save_frame(frame: np.ndarray, filename: str):
    """Save a single frame as a PNG using OpenCV."""
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr_frame)    

def write_video(observations: List[np.ndarray], filename: str, fps: int = 15):
    """Save a sequence of numpy arrays as a video using OpenCV."""
    height, width, layers = observations[0].shape
    size = (width, height)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for obs in observations:
        bgr_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()    

def frames_from_rendered_pixels(demos, output_directory, traj_base_names = None, name_prefix=None):
    '''
    Process trajectories to get videos and first/last frames from rendered pixels.
    trajectories: list of MAGICALTrajectory objects
    output_directory: directory to save videos and frames
    traj_base_names: list of base names for trajectories (optional)
    name_prefix: prefix to add to filenames, will be used if traj_base_names is None.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if traj_base_names:
        assert len(demos) == len(traj_base_names)

    def save_first_last_frames(observations: List[np.ndarray], prefix: str):
        first_frame_filename = os.path.join(output_directory, f'frame-{prefix}-{base_filename}-first.png')
        last_frame_filename = os.path.join(output_directory, f'frame-{prefix}-{base_filename}-last.png')
        save_frame(observations[0], first_frame_filename)
        save_frame(observations[-1], last_frame_filename)
    
    for idx, demo in enumerate(demos):
        env_name = demo['env_name']
        traj = demo['trajectory']
        obs_sequence= traj.obs
        egocentric_views = [obs['ego'] for obs in obs_sequence]
        allocentric_views = [obs['allo'] for obs in obs_sequence]
        # Concatenate egocentric and allocentric views side by side
        concat_views = [np.concatenate((allo, ego), axis=1) for allo, ego in zip(allocentric_views, egocentric_views)]

        if traj_base_names:
            base_filename = traj_base_names[idx]
        # Extract base filename without path and extension
        elif name_prefix:
            base_filename = name_prefix + str(idx)
        else:
            now = datetime.datetime.now()
            time_str = now.strftime('%FT%H:%M:%S')
            base_filename = f"{env_name}-{time_str}-{idx}"
        # Save first and last frames for each view
        save_first_last_frames(egocentric_views, "ego")
        save_first_last_frames(allocentric_views, "allo")
        save_first_last_frames(concat_views, "concat")
    
        # Write the videos
        write_video(egocentric_views, os.path.join(output_directory, f'video-ego-{base_filename}.mp4'))
        write_video(allocentric_views, os.path.join(output_directory, f'video-allo-{base_filename}.mp4'))
        write_video(concat_views, os.path.join(output_directory, f'video-concat-{base_filename}.mp4'))
