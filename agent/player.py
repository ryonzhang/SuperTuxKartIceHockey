import numpy as np
from chris_test.models import Detector, load_model
from .controller import Controller1
from collections import deque
from torchvision.transforms import functional as F
import pystk

class History:
    def __init__(self, max_history_length, default):
        self.elements = []
        self.max_history_length = max_history_length
        self.default = default

    def push(self, e):
        if len(self.elements) < self.max_history_length:
            self.elements.append(e)
        elif self.elements[0] < e:
            del self.elements[0]
            self.elements.append(e)
    def peek(self, e):
        if (len(self.elements)!=0):
            return self.elements[-1]
        return self.default

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """

    kart = "wilber"

    # A static history data structure, this is accessed in the controller
    last_seen_q = History(max_history_length = 20, default = np.array([0.0,0.0]))
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.team = player_id % 2
        self.team_orientaion_multiplier = -2*(self.team%2)+1
        self.model = load_model()
        self.player_id = player_id//2

       
        self.controller = Controller1(self.team_orientaion_multiplier,self.player_id)

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        print(player_info.kart.location)
        # This returns the puck location if we can see the puck
        # last_seen_side is: -1 is left, 1 is right
        # puck_location is: None if we cant see the puck, [x, z]
        # self.team_orientaion_multiplier is a multiplier to any game position argument

        last_seen_side, puck_location = self.model.detect(F.to_tensor(image), player_info)

        action = self.controller.act(action, player_info, puck_location, last_seen_side)

        return action


