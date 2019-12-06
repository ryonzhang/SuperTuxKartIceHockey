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
