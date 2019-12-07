import numpy as np
import torch
DEBUG = False

class History:
    def __init__(self, max_history_length, default):
        self.elements = []
        self.max_history_length = max_history_length
        self.default = default

    def push(self, e):
        if len(self.elements) < self.max_history_length:
            self.elements.append(e)
        else:
            del self.elements[0]
            self.elements.append(e)
    def peek(self, N=1):
        if (len(self.elements)>=N):
            return self.elements[-N:]
        elif (len(self.elements)!=0):
            return self.elements[:-1]
        return self.default
        


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
    goalieID=0.

    

    def __init__(self, team_orientaion_multiplier, player_id):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        self.team_orientaion_multiplier = team_orientaion_multiplier
        self.player_id = player_id
        self.goal = np.array([0.0,64.5]) #const
        self.goalKeepLoc = np.array([0.0,-66]) #const
        self.attempted_to_fire = False
        self.his = History(max_history_length = 10, default = np.array([0.0,0.0]))
        self.last_world_pos = np.array([0.0,-64.5])
     
    def act(self, action, player_info, puck_location=None, last_seen_side=None, testing=False):
        # Fire every other frame
        action["fire"]= self.attempted_to_fire 
        self.attempted_to_fire = not self.attempted_to_fire
        # Get world positions
        if (testing and puck_location is not None):
            #  Standardizing direction 2 elements
            # [0] is negitive when facing left side of court (left of your goal), positive when right
            # [1] is positive towards enemy goal, negitive when facing your goal
            puck_location*=self.team_orientaion_multiplier

        pos_me = to_numpy(player_info.kart.location)*self.team_orientaion_multiplier
        
        # Get kart vector
        front_me = to_numpy(player_info.kart.front)*self.team_orientaion_multiplier
        ori_me = get_vector_from_this_to_that(pos_me, front_me)

        # Determine we are moving backwards
        backing_turn_multiplier = 1.
        kart_vel = np.dot(to_numpy(player_info.kart.velocity)*self.team_orientaion_multiplier,ori_me)
        if kart_vel < 0:
            backing_turn_multiplier = -1.
        if DEBUG:
            print("kart_vel",kart_vel)

        # determine if we are  in a new round
        if (kart_vel == 0 and abs(np.linalg.norm(self.last_world_pos-pos_me))>5):
            Controller1.goalieID=0.

        self.last_world_pos = pos_me
            
        if (Controller1.goalieID==self.player_id): # I'm goalie
            ori_to_goalKeepLoc = get_vector_from_this_to_that(pos_me, self.goalKeepLoc,normalize=False)
            ori_to_goalKeepLoc_n = get_vector_from_this_to_that(pos_me, self.goalKeepLoc)
            to_goalKeepLoc_mag = np.linalg.norm(ori_to_goalKeepLoc)
            if DEBUG:
                print("goalie_mag", to_goalKeepLoc_mag)
            if puck_location is not None and (puck_location[1]<-24.5): # puck is in our third, Attack!
                self.his.push(puck_location)
                ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
                ori_puck_to_goal = get_vector_from_this_to_that(puck_location, self.goal,normalize=False)
                ori_puck_to_goal_n = get_vector_from_this_to_that(puck_location, self.goal,normalize=True)
                

                action["acceleration"] = 1

                _his = self.his.peek(2)
                if len(_his)==2 and get_vector_from_this_to_that(_his[0],_his[1],normalize=False)[1]<-1:
                    dif = get_vector_from_this_to_that(_his[0],_his[1],normalize=False)
                    if DEBUG:
                        print("dif",dif)
                    pos_hit_loc = puck_location + 2*dif
                else:
                    pos_hit_loc = puck_location-.5*ori_puck_to_goal_n
                
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)
                turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
                if turn_mag > 1e-25:
                    action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*50000*backing_turn_multiplier
                if DEBUG:
                    print("trying to switch",abs(angle_between(ori_to_puck,ori_me)) < .8, puck_location[1]>-35, np.linalg.norm(ori_to_puck)<15,np.linalg.norm(ori_to_puck))
                if (abs(angle_between(ori_to_puck,ori_me)) < .8 and puck_location[1]>-35 and np.linalg.norm(ori_to_puck)<15): #switch rolls
                    if DEBUG:
                        print("SWITCHING",Controller1.goalieID)
                    Controller1.goalieID = Controller1.goalieID+1%2
            elif (to_goalKeepLoc_mag>2): #Goalie isnt at goal keeper location
                if np.dot(ori_to_goalKeepLoc,ori_me)<0:
                    action["brake"] = 1.
                    action["acceleration"] = 0.0
                else:
                    action["acceleration"] = 0.2
                turn_mag = abs(1 - np.dot(ori_me, ori_to_goalKeepLoc_n))
                #print(turn_mag)
                if turn_mag > 1e-25:
                    action["steer"] = -1*np.sign(np.cross(ori_to_goalKeepLoc_n, ori_me))*turn_mag*5000*backing_turn_multiplier
            elif (to_goalKeepLoc_mag<8): # At goal keeper location
                if kart_vel < 0:
                    action["acceleration"] = abs(kart_vel/10)
                if puck_location is not None:
                    self.his.push(puck_location)
                    ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                    ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
                    
                    
                    turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
                    
                    if turn_mag > .0005:
                        if DEBUG:
                            print("turn_mag",turn_mag)
                        if np.dot(ori_to_goalKeepLoc,ori_me)<0:
                            action["brake"] = 1.
                            action["acceleration"] = 0.0
                        else:
                            action["acceleration"] = (abs(kart_vel)+.2)/4.5
                        action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000*backing_turn_multiplier
        
                else: # Doesn't have vision of puck
                    if DEBUG:
                        print("last_seen_side",last_seen_side)
                    action["brake"] = 1.
                    action["acceleration"] = 0.0
                    action["steer"] = backing_turn_multiplier*last_seen_side
        else: # I'm Striker
            if puck_location is not None:

                ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
                ori_puck_to_goal = get_vector_from_this_to_that(puck_location, self.goal,normalize=False)
                ori_puck_to_goal_n = get_vector_from_this_to_that(puck_location, self.goal,normalize=True)

                to_puck_mag = np.linalg.norm(ori_to_puck)
                #if (pos_me[1]>24.5): #there third
                #elif (pos_me[1]<-24.5): #our third
                
                if (to_puck_mag>20): # not close to puck
                    action["acceleration"] = 1
                    if (to_puck_mag>80):# really far
                        action["acceleration"] = .8
                    if DEBUG:
                        print("not close to puck")
                    turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
                    #print(turn_mag)
                    if turn_mag > 1e-25:
                        action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000*backing_turn_multiplier
                else: # close to puck
                    if DEBUG:
                        print("close to puck")
                    #ab_player_puck = angle_between(ori_to_puck,ori_me)

                    action["acceleration"] = .8
                    if (to_puck_mag>10):# really close
                        action["acceleration"] = .5
                    pos_hit_loc = puck_location-1.3*ori_puck_to_goal_n
                    ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)
                    turn_mag = abs(1 - np.dot(ori_me, ori_to_puck_n))
                    if turn_mag > 1e-25:
                        action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_mag*5000*backing_turn_multiplier

            else: # Doesn't have vision of puck
                if DEBUG:
                    print("last_seen_side",last_seen_side)
                action["brake"] = 1.
                action["acceleration"] = 0.0
                action["steer"] = backing_turn_multiplier*last_seen_side

        return action

