import numpy as np
import torch
from collections import deque # for queue

RADIAN_TO_DEGREE = 180/np.pi
memory = deque(maxlen=3)

class Controller1:
    def __init__(self, team):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        self.team = team
        self.team_orientaion_multiplier = -2*(team%2)+1

    
    def act(self, action, player_info ,puck_location_onscreen=None):
        # angle = (np.arctan2(aim_point[0], aim_point[2]) * 180.0 / np.pi)
        # will add the piazza code from screen to world found below


        # a = angle

        # memory.append(a)
        # if sum(memory) / len(memory) > 25: # smoothing , previously just above abs(a) > 35:
        #     action.drift=True
        # else:
        #     action.drift=False
        # action.steer = a
        # action.acceleration = 1.0 if current_vel < 22.0 else 0.0
        return action



# https://piazza.com/class/jzsr2l6kqaa5xi?cid=395

# SCREEN_WIDTH = 400
# SCREEN_HEIGHT = 300

# def ray_trace_to_ground(numpy_vec4, view):
#     assert view.shape == (4, 4)
#     assert numpy_vec4.shape[0] == 4
#     assert abs(numpy_vec4[3]) < 1e-7

#     camera_location = np.array(list(np.linalg.pinv(view)[:3,3]) + [0])
#     ground_y_coord = 0.3698124289512634,
#     multiplier = (ground_y_coord - camera_location[1]) / numpy_vec4[1]
#     result = camera_location + multiplier * numpy_vec4
#     return result

# def view_to_global(numpy_vec4, view):
#     assert numpy_vec4.shape[0] == 4
#     view_inverse = np.linalg.pinv(view)
#     return view_inverse @ numpy_vec4

# def homogeneous_to_euclidean(numpy_vec4):
#     assert numpy_vec4.shape[0] == 4
#     # dont want numerical errors to magnify... prolly dont need this check but whatever
#     if abs(numpy_vec4[3]) <= 1e-4:
#         result[3] = 0
#         return numpy_vec4
#     result = numpy_vec4 / numpy_vec4[3]
#     result[3] = 0
#     return result

# def screen_to_view(aim_point_image, proj, view):
#     x, y, W, H = *aim_point_image, SCREEN_WIDTH, SCREEN_HEIGHT
#     projection_inverse = np.linalg.pinv(proj)
#     ndc_coords = np.array([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1])
#     return projection_inverse @ ndc_coords

# def screen_puck_to_world_puck(screen_puck_coords, proj, view):
#     """
#     Call this function with
#     @param screen_puck_coords: [screen_puck.x, scren_puck.y]
#     @param proj: camera.projection.T
#     @param view: camera.view.T
#     """
#     view_puck_coords = homogeneous_to_euclidean(screen_to_view(screen_puck_coords, proj, view))
#     view_puck_dir = view_puck_coords / np.linalg.norm(view_puck_coords)
#     global_puck_dir = view_to_global(view_puck_dir, view)
#     global_puck_dir = global_puck_dir / np.linalg.norm(global_puck_dir)
#     return ray_trace_








if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    # parser.add_argument('-m', '--max_frames', type=int, default=3)
    args = parser.parse_args()
    test_controller(args)
