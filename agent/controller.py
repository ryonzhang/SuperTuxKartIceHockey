import numpy as np
import torch

class Controller1:
    def __init__(self, team):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        self.team = team
        self.team_orientaion_multiplier = -2*(team%2)+1

    
    def act(self, action, player_info ,puck_location_onscreen=None):

        return action
