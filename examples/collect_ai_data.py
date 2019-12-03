# Records actions, positions of puck and all players in world and screen space for 4 ai players
# use argument --display to show on screen, --steps to specify number of frames to save

# sets all players to random karts so network can recognize all of them (maybe run this multiple times so it can see all of them once)

# saves data from player i in drive_data/[player nr]
# files:
# .png: game video output
# .npz: numpy arrays of positions of team 0,1 players and ball in separate files;
#   world positions in world directory (position on map) and on-screen positions of the four players in respective dirs
# .txt: last action player took
# rest: dump of player state, depth, segmentation map

# goal: use this to collect data
# -create net to predict onscreen positions of karts, ball from game video output
# -calculate world map from those predictions (code for that can be found on piazza (https://piazza.com/class/jzsr2l6kqaa5xi?cid=395))
#    combine sight of both players into map, use some clever logic to also predict objects that neither of the players can see
# - create net that can take actions from map (imitate ai first (this script saves player actions and all relevant positions), gradient free optimization later)
#    note for gradient free: distance of puck to goal could be a good reward function (e.g. reward = 1/(distance+1))
#    note for second net in general: map coordinates need to be rotated by 180 degrees (not flipped, else steering left/right are also flipped) depending on which team we're on
#       e.g. if our id is 1, we have to regard team 1 as our own (-> maybe flip heatmaps or something) and aim for the other goal, else we would have to train a red agent and a blue agent

from pathlib import Path
from PIL import Image
import argparse
import pystk
from time import time
import numpy as np
import _pickle as pickle
import random
import uuid
import os
from . import gui

# ripped from hw6
def to_image(x, proj, view):
    W, H = 400, 300
    p = proj @ view @ np.array(list(x) + [1])
    return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

def to_numpy(location):
    return np.float32([location[0], location[1], location[2]])

def action_dict(action):
    return {k: getattr(action, k) for k in ['acceleration', 'brake', 'steer', 'fire', 'drift']}


if __name__ == "__main__":
    # create uuid for file names
    uid = str(uuid.uuid1())

    soccer_tracks = {"soccer_field", "icy_soccer_field"}

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track', default = 'icy_soccer_field')
    parser.add_argument('-k', '--kart', default='')
    parser.add_argument('--team', type=int, default=0, choices=[0, 1])
    parser.add_argument('-s', '--step_size', type=float)
    parser.add_argument('-v', '--visualization', type=str, choices=list(gui.VT.__members__), nargs='+',
                        default=['IMAGE'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_dir', type=Path, default = 'drive_data')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--steps', type=int, default=10000)

    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        # create dirs
        if not os.path.exists(args.save_dir / 'world'):
            os.makedirs(args.save_dir / 'world')
        if not os.path.exists(args.save_dir / '0'):
            os.makedirs(args.save_dir / '0')
        if not os.path.exists(args.save_dir / '1'):
            os.makedirs(args.save_dir / '1')
        if not os.path.exists(args.save_dir / '2'):
            os.makedirs(args.save_dir / '2')
        if not os.path.exists(args.save_dir / '3'):
            os.makedirs(args.save_dir / '3')

    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    possible_karts = ['tux', 'gnu', 'nolok', 'sara', 'adiumy', 'konqi', 'kiki', 'beastie',
        'amanda', 'emule', 'suzanne', 'gavroche', 'hexley', 'xue', 'pidgin', 'puffy', 'wilber']

    config = pystk.RaceConfig()
    config.num_kart = 4

    config.difficulty = 2

    num_player = 4
    config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL
    for i in range(3):
        config.players.append(
                pystk.PlayerConfig(random.choice(possible_karts), pystk.PlayerConfig.Controller.AI_CONTROL, (args.team + i + 1) % 2))

    config.players[0].team = args.team


    for p in config.players:
        p.kart = random.choice(possible_karts)

    if args.track is not None:
        config.track = args.track
        if args.track in soccer_tracks:
            config.mode = config.RaceMode.SOCCER
    if args.step_size is not None:
        config.step_size = args.step_size

    race = pystk.Race(config)
    race.start()

    if (args.display):
        uis = [gui.UI([gui.VT[x] for x in args.visualization]) for i in range(num_player)]
    save_depth = "DEPTH" in args.visualization
    save_labels = "SEMANTIC" in args.visualization or "INSTANCE" in args.visualization

    state = pystk.WorldState()
    t0 = time()
    n = 0
    while (n<args.steps) and ((not args.display) or all(ui.visible for ui in uis)):
        if (not args.display) or (not all(ui.pause for ui in uis)):
            #race.step(uis[0].current_action)
            race.step()
            state.update()
            if args.verbose and config.mode == config.RaceMode.SOCCER:
                print('Score ', state.soccer.score)
                print('      ', state.soccer.ball)
                print('      ', state.soccer.goal_line)

        if(args.display):
            for ui, d in zip(uis, race.render_data):
                ui.show(d)

        if args.save_dir:

            # save positions of karts, ball
            pos_ball = to_numpy(state.soccer.ball.location)
            pos_kart_0 = to_numpy(state.karts[0].location)
            pos_kart_1 = to_numpy(state.karts[1].location)
            pos_kart_2 = to_numpy(state.karts[2].location)
            pos_kart_3 = to_numpy(state.karts[3].location)

            np.savez(args.save_dir / 'world' / (uid + '_pos_ball_%06d' % n), pos_ball)
            np.savez(args.save_dir / 'world' / (uid + '_pos_team0_%06d' % n), pos_kart_0, pos_kart_2)
            np.savez(args.save_dir / 'world' / (uid + '_pos_team1_%06d' % n), pos_kart_1, pos_kart_3)


            # save positions, actions for all players
            for i in range(len(race.render_data)):
                image = np.array(race.render_data[i].image)
                action = race.last_action[i]#action_dict(uis[i].current_action)
                player_info = state.karts[i]

                # get kart, ball positions on screen of this player
                proj = np.array(state.players[i].camera.projection).T
                view = np.array(state.players[i].camera.view).T
                local_ball = to_image(pos_ball, proj, view)
                local_kart_0 = to_image(pos_kart_0, proj, view)
                local_kart_1 = to_image(pos_kart_1, proj, view)
                local_kart_2 = to_image(pos_kart_2, proj, view)
                local_kart_3 = to_image(pos_kart_3, proj, view)

                # save to files
                np.savez(args.save_dir / str(i) / (uid + '_pos_ball_%06d' % n), local_ball)
                np.savez(args.save_dir / str(i) / (uid + '_pos_team0_%06d' % n), local_kart_0, local_kart_2)
                np.savez(args.save_dir / str(i) / (uid + '_pos_team1_%06d' % n), local_kart_1, local_kart_3)

                Image.fromarray(image).save(args.save_dir / str(i) / (uid + '_image_%06d.png' % n))
                (args.save_dir / str(i) / (uid + '_action_%06d.txt' % n)).write_text(str(action))
                with open(args.save_dir / str(i) / (uid + '_player_info_%06d' % n), 'wb') as output:
                    pickle.dump(player_info, output, -1)
                if save_depth:
                    depth = np.array(race.render_data[i].depth).astype('uint8')
                    np.save(args.save_dir / str(i) / (uid + '_depth_%06d' % n), depth)
                if save_labels:
                    label = np.array(race.render_data[i].instance) #& 0xffffff
                    np.save(args.save_dir / str(i) / (uid + '_label_%06d' % n), label)

        # Make sure we play in real time
        n += 1
        if(args.display):
            delta_d = n * config.step_size - (time() - t0)
            if delta_d > 0:
                ui.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
