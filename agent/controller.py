import numpy as np
import torch

def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])


def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Controller1:
    def __init__(self, team_orientaion_multiplier, player_id):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        self.team_orientaion_multiplier = team_orientaion_multiplier
        self.player_id = player_id
        self.goal = np.array([0.0,64.5])

    
    def act(self, action, player_info, puck_location=None, last_seen_side=None, testing=False):
        if puck_location is not None:
            pos_me = to_numpy(player_info.location)
            if (testing):
                #  Standardizing direction 2 elements
                # [0] is negitive when facing left side of court (left of your goal), positive when right
                # [1] is positive towards enemy goal, negitive when facing your goal
                puck_location*=self.team_orientaion_multiplier
            pos_me*=self.team_orientaion_multiplier


            # Get some directional vectors. 
            front_me = to_numpy(player_info.front)*self.team_orientaion_multiplier
            ori_me = get_vector_from_this_to_that(pos_me, front_me)
            ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
            ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
            ori_puck_to_goal = get_vector_from_this_to_that(puck_location, self.goal,normalize=False)

            

            # Turn towards the item to pick up. Not very good at turning.
            action["acceleration"] = 0.5
            to_puck_mag = np.linalg.norm(ori_to_puck)
            print("ori_to_puck",ori_to_puck,to_puck_mag)

            #if (to_puck_mag>10):
            turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
            #print(turn_mag)
            if turn_mag > 1e-25:
                action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000
            

        elif last_seen_side is not None:
            print("last_seen_side",last_seen_side)



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
