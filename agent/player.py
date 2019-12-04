import numpy as np
from .models import Detector, load_model
from .controller import Controller1

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
        self.model = load_model("det.th")
        self.player_id = player_id//2

        if self.player_id == 0: # Should be the goalie TODO:Figure out if the goalie id should be 0 or 1
            self.controller = Controller1(self.team)
        else: # Attacker(s), only one attacker for 2v2
            self.controller = Controller1(self.team)

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        # We might want to pass in player_info
        # puck_location_onscreen == None when the puck isn't on the screen
        puck_location_onscreen = self.model(image)

        action = self.controller.act(action, player_info, puck_location_onscreen)

        return action


