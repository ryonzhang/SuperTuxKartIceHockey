import pystk
import numpy as np
import torch



def control1(action, is_puck_onscreen, puck_location_onscreen, team, player_info):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in local coordinate frame
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    return action