import numpy as np
from .models import Detector
from .controller import control1
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
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.team = player_id % 2
        self.model = Detector.load_model("det.th")
        self.player_id = player_id//2
        pass
        
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
        
        
        if self.player_id == 0: # Should be the goalie TODO:Figure out if the goalie id should be 0 or 1
            is_puck_onscreen, puck_location_onscreen = self.model(image) # We might want to pass in player_info
            action = control1(action, is_puck_onscreen, puck_location_onscreen, self.team, player_info)
        else: # Attacker(s), only one attacker for 2v2
            is_puck_onscreen, puck_location_onscreen = self.model(image) # We might want to pass in player_info
            action = control1(action, is_puck_onscreen, puck_location_onscreen, self.team, player_info)

        return action

