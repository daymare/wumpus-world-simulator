# PyAgent.py

import random

#import tensorflow as tf
import numpy as np

import Action
import Orientation



class Util():
    @staticmethod
    def get_random_lr():
        num = random.randint(0, 1)

        if num == 0:
            return Action.TURNLEFT
        else:
            return Action.TURNRIGHT

    @staticmethod
    def add_row_seperator(display_map, ypos, cell_width):
        plus_distance = 0 # distance until we need to 
            # display another plus

        # TODO double check shape is correct here
        for xpos in range(display_map.shape[0]):
            if plus_distance == 0:
                display_map[xpos, ypos] = '+'
                plus_distance = cell_width
            else:
                display_map[xpos, ypos] = '-'
                plus_distance -= 1

class Map:
    """ class for keeping track of the world map at any given time
        
        position vector contains:
        indecies:
            0 - OK
            1 - Visited
            2 - possible_pit
            3 - possible_wumpus
            4 - stench
            5 - breeze
            6 - glitter
            7 - pit
            8 - wumpus
        
    """
    def __init__(self):
        self.size_x = 4
        self.size_y = 4

        vector_dim = 9
        world_map = np.zeros((self.size_x, self.size_y, vector_dim))

        self.index_map = {
            "ok" : 0,
            "visited" : 1,
            "possible_pit" : 2,
            "possible_wumpus" : 3,
            "stench" : 4,
            "breeze" : 5,
            "glitter" : 6,
            "pit" : 7,
            "wumpus" : 8
            }

    def print(self):
        """ print the map to the screen

            each cell will be represented by numpy arrays
            each cell will then be stitched together to form the map
            then we print out the map
        """
        # note that symbols cannot be numbers
        # TODO validate symbol map
        symbol_map = \
            {
                '0' : "ok",
                '1' : "V",
                '2' : "PP",
                '3' : "PW",
                '4' : "S",
                '5' : "B",
                '6' : "G",
                '7' : "P",
                '8' : "W"
            }
        cell_layout = \
            [
                ['', '0', '', '1', ''],
                ['', '4', '', '5', ''],
                ['', '7', '', '8', ''],
                ['', '', '', '', ''],
                ['-', '-', '-', '-', '-'],
                ['', '2', '', '3', '']
            ]
        cell_layout = np.array(cell_layout, dtype=np.character)
        cell_layout = np.transpose(cell_layout)

        # TODO double check cell layout shape is as expected

        cell_width = cell_layout.shape[0]
        cell_height = cell_layout.shape[1]

        def build_cell(map_x, map_y):
            """ build a display cell from the given map indecies
            """
            position = self.world_map[map_x, map_y]

            xdim = cell_width
            ydim = cell_height

            cell = cell_layout.copy()

            for x in range(xdim):
                for y in range(ydim):
                    if cell[x, y] in symbol_map:
                        indicator = cell[x, y]
                        symbol = symbol_map[indicator]

                        # replace this character in the symbol map
                        cell[x, y] = ''

                        # check if the symbol exists at this position
                        symbol_index = int(indicator)
                        if position[symbol_index] == 1:
                            for i in range(len(symbol)):
                                cell[x+i, y] = symbol[i]
            
            return cell
        def build_map(cells):
            """ build a display map from a matrix of cells
                dimensions should be (
                    1 + (cell_width + 1) * num_cellsx, 
                    1 + (cell_height + 1) * num_cellsy)
            """
            # TODO double check shape of world map is as expected
            map_width = self.world_map.shape[0]
            map_height = self.world_map.shape[1]

            xdim = 1 + (cell_width + 1) * map_width
            ydim = 1 + (cell_height + 1) * map_height

            display_map = np.full((xdim, ydim), '')
            xpos = 0
            ypos = 0

            # set first spacer
            Util.add_row_seperator(display_map, 0, cell_width)
            
            for y in range(map_height):
                # set the cells

                # set the next spacer
                pass
            



    def update(self, x, y, percept):
        # ensure current position is set visited and OK
        world_map[x, y, self.index_map["ok"]] = 1
        world_map[x, y, self.index_map["visited"]] = 1

        # handle glitter
        if percept["glitter"] is True:
            world_map[x, y, self.index_map["glitter"]] = 1

        # handle stenches
        if percept["stench"] is True:
            world_map[x, y, self.index_map["stench"]] = 1
            self._update_neighbors(x, y, "stench")
        else:
            self._update_neighbors(x, y, "no_stench")

        # handle breezes
        if percept["breeze"] is True:
            world_map[x, y, self.index_map["breeze"]] = 1
            self._update_neighbors(x, y, "breeze")
        else:
            self._update_neighbors(x, y, "no_breeze")


    def get_pos(self, x, y):
        return world_map[x, y]

    def get_flat_map(self):
        return world_map.flatten()

    def _update_neighbors(self, x, y, value):
        # TODO handle logical inference on pits and wumpus
        for pos in self._get_neighbors(x, y):
            cx = pos[0]
            cy = pos[1]

            if value == "stench":
                if world_map[cx, cy, self.index_map["ok"]] != 1:
                    world_map[cx, cy, self.index_map["possible_wumpus"]] = 1
            elif value == "breeze":
                if world_map[cx, cy, self.index_map["ok"]] != 1:
                    world_map[cx, cy, self.index_map["possible_pit"]] = 1
            elif value == "no_breeze":
                world_map[cx, cy, self.index_map["possible_pit"]] = 0
                if world_map[cx, cy, self.index_map["possible_wumpus"]] == 0:
                    world_map[cx, cy, self.index_map["ok"]] = 1
            elif value == "no_stench":
                world_map[cx, cy, self.index_map["possible_wumpus"]] = 0
                if world_map[cx, cy, self.index_map["possible_pit"]] == 0:
                    world_map[cx, cy, self.index_map["ok"]] = 1

    def _get_neighbors(self, x, y):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        if x > 0:
            min_x = -1
        if x < self.size_x:
            max_x = 1
        if y > 0:
            min_y = -1
        if y < self.size_y:
            max_y = 1

        for nx in range(x - minx, x + max_x + 1):
            if nx != x:
                yield (nx, y)
        for ny in range(y - min_y, y + max_y + 1):
            if ny != y:
                yield (x, ny)

class Agent:
    """ pyagent wrapper class
    """
    def __init__(self):
        self.last_action = None
        self.x = None
        self.y = None
        self.direction = None
        self.hasgold = None

    def destructor(self):
        pass

    def initialize(self):
        self.last_action = None
        self.x = 1
        self.y = 1
        self.direction = "east"
        self.hasgold = False
        self.hasarrow = True

    def process(self, stench, breeze, glitter, bump, scream):
        """ process the percept and return desired action

        input:
            stench - boolean, whether stench is detected
            breeze - boolean, whether breeze is detected
            glitter - boolean, whether glitter is detected
            bump - boolean, whether bump has resulted from the last action
            scream - boolean, whether scream has resulted from the last action
        """
        # update location
        if self.last_action == Action.GOFORWARD and bump is False:
            self.update_location()

        current_action = None

        # part a: if glitter then grab gold
        if glitter is True:
            self.hasgold = True
            current_action = Action.GRAB

        #  part b: if can win then win
        elif self.x == 1 and \
            self.y == 1 and \
            self.hasgold is True:
            current_action = Action.CLIMB

        # part c: if stench and arrow then shoot
        elif stench is True and self.hasarrow is True:
            self.hasarrow = False
            current_action = Action.SHOOT

        # part d: if bump then left or right
        elif bump is True:
            current_action = Util.get_random_lr()
            self.update_orientation(current_action)

        # part e: if nothing else go straight
        else:
            current_action = Action.GOFORWARD

        self.last_action = current_action
        return current_action

    def gameover(self):
        pass

    def update_location(self):
        """ update the location on goforward action
            
            move action must have succeeded.
            This function does not attempt to check if
            the move is valid
        """
        x = self.x
        y = self.y
        direction = self.direction

        if direction == "north":
            y = y + 1
        elif direction == "south":
            y = y - 1
        elif direction == "east":
            x = x + 1
        else: # west
            x = x - 1

        self.x = x
        self.y = y

    def update_orientation(self, turn_action):
        assert turn_action == Action.TURNLEFT or \
                turn_action == Action.TURNRIGHT

        current_direction = self.direction
        
        if turn_action == Action.TURNLEFT:
            if current_direction == "north":
                current_direction = "west"
            elif current_direction == "south":
                current_direction = "east"
            elif current_direction == "east":
                current_direction = "north"
            else: # west
                current_direction = "south"
        else: # turning right
            if current_direction == "north":
                current_direction = "east"
            elif current_direction == "south":
                current_direction = "west"
            elif current_direction == "east":
                current_direction = "south"
            else: # west
                current_direction = "north"

        self.direction = current_direction

myagent = None



# Input functions
def PyAgent_Constructor ():
    global myagent
    myagent = Agent()

def PyAgent_Destructor ():
    global myagent
    if myagent is None:
        raise Exception("cannot call destructor when there is no agent")
    myagent.destructor()
    myagent = None

def PyAgent_Initialize ():
    global myagent
    if myagent is None:
        PyAgent_Constructor()
    myagent.initialize()

def PyAgent_Process (stench,breeze,glitter,bump,scream):
    global myagent
    if myagent is None:
        raise Exception("cannot call process when there is no agent")
    stench = bool(stench)
    breeze = bool(breeze)
    glitter = bool(glitter)
    bump = bool(bump)
    scream = bool(scream)
    return myagent.process(stench, breeze, glitter, bump, scream)

def PyAgent_GameOver (score):
    global myagent
    if myagent is None:
        raise Exception("cannot call gameover when there is no agent")
    myagent.gameover()
