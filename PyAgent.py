# PyAgent.py

import random
import heapq
import copy

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
    def add_row_separator(display_map, ypos, cell_width, border=False):
        plus_distance = 0 # distance until we need to 
            # display another plus

        # TODO double check shape is correct here
        for xpos in range(display_map.shape[0]):
            if plus_distance == 0:
                display_map[xpos, ypos] = '+'
                plus_distance = cell_width
            else:
                if border is False:
                    display_map[xpos, ypos] = '-'
                else:
                    display_map[xpos, ypos] = '='
                plus_distance -= 1

    @staticmethod
    def extract_cell_row(cells, ypos):
        # cells shape should be  (xdim, ydim)
        xdim = cells.shape[0]
        cell_row = []
        for x in range(xdim):
            cell_row.append(cells[x, ypos])

        return cell_row

    @staticmethod
    def add_cell_row(display_map, starting_ypos, cells_y, cells,
            cell_width, cell_height, border=-1):
        ypos = starting_ypos

        cell_row = Util.extract_cell_row(cells, cells_y)
        xpos = 0

        for ydiff in range(cell_height):
            xpos = 0

            # add leftmost separator
            display_map[xpos, ypos] = '|'
            xpos += 1

            cell_x = 0
            for cell in cell_row:
                for xdiff in range(cell_width):
                    display_map[xpos, ypos] = cell[xdiff, ydiff]
                    xpos += 1

                # add right row separator
                if cell_x != border - 1:
                    display_map[xpos, ypos] = '|'
                else:
                    display_map[xpos, ypos] = ']'

                xpos += 1
                cell_x += 1
            ypos += 1

        final_y = ypos

        return final_y
        
    @staticmethod
    def get_direction(loc0, loc1):
        """ loc0 and loc1 are tuples of (x, y)

            loc 1 is desired location
        """
        xdiff = loc1[0] - loc0[0]
        ydiff = loc1[1] - loc0[1]

        # one and only one of these should be nonzero
        assert (xdiff != 0) != (ydiff != 0)

        if xdiff > 0:
            return "east"
        elif xdiff < 0:
            return "west"
        elif ydiff > 0:
            return "north"
        elif ydiff < 0:
            return "south"
        raise Exception("how did we get here?!")

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
        self.found_borders = False

        self.seen_scream = False
        self.found_wumpus = False

        self.wumpus_loc = None
        self.gold_loc = None

        self.vector_dim = vector_dim = 9
        self.world_map = np.zeros((self.size_x + 2, self.size_y + 2, 
            vector_dim), dtype=np.int)

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


    def reset(self):
        """ reset the map for the next runthrough

            basically just props the wumpus back up

        """
        if found_wumpus is True:
            x, y = self.wumpus_loc
            self.set(x, y, "wumpus", 1)
            self.set(x, y, "ok", 0)
            self.set(x, y, "visited", 0)

    def print(self):
        """ print the map to the screen

            each cell will be represented by numpy arrays
            each cell will then be stitched together to form the map
            then we print out the map
        """
        # note that symbols cannot be numbers
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
        negated_symbol_map = \
            {
                '2' : "NP",
                '3' : "NW"
            }
        cell_layout = \
            [
                [' ', '0', ' ', ' ', '1', ' ', ' '],
                [' ', '4', ' ', ' ', '5', ' ', ' '],
                [' ', '7', ' ', ' ', '8', ' ', ' '],
                [' ', '2', ' ', ' ', '3', ' ', ' ']
            ]
        cell_layout = np.array(cell_layout, dtype=np.character)
        cell_layout = np.transpose(cell_layout)

        # TODO double check cell layout shape is as expected

        cell_width = cell_layout.shape[0]
        cell_height = cell_layout.shape[1]
        map_width = self.world_map.shape[0]
        map_height = self.world_map.shape[1]

        # TODO double check build cell works as expected
        def build_cell(map_x, map_y):
            """ build a display cell from the given map indecies
            """
            position = self.world_map[map_x, map_y]

            xdim = cell_width
            ydim = cell_height

            cell = cell_layout.copy()

            for x in range(xdim):
                for y in range(ydim):
                    if cell[x, y].decode("utf-8") in symbol_map:
                        indicator = cell[x, y].decode("utf-8")
                        symbol = symbol_map[indicator]
                        if indicator in negated_symbol_map:
                            negated_symbol = negated_symbol_map[indicator]

                        # replace this character in the symbol map
                        cell[x, y] = ' '

                        # check if the symbol exists at this position
                        symbol_index = int(indicator)
                        if position[symbol_index] == 1:
                            for i in range(len(symbol)):
                                cell[x+i, y] = symbol[i]
                        elif position[symbol_index] == -1:
                            for i in range(len(negated_symbol)):
                                cell[x+i, y] = negated_symbol[i]
            
            return cell

        # TODO double check build map works as expected
        def build_map(cells):
            """ build a display map from a matrix of cells
                dimensions should be (
                    1 + (cell_width + 1) * num_cellsx, 
                    1 + (cell_height + 1) * num_cellsy)
            """
            # TODO double check shape of world map is as expected

            xdim = 1 + (cell_width + 1) * map_width
            ydim = 1 + (cell_height + 1) * map_height

            display_map = np.full((xdim, ydim), '')
            xpos = 0
            ypos = 0

            # set first spacer
            Util.add_row_separator(display_map, ypos, cell_width, border=False)
            ypos += 1
            
            for y in range(map_height):
                # set the cells
                ypos = Util.add_cell_row(display_map, ypos, y, cells,
                        cell_width, cell_height, border=self.size_x)

                # set the next spacer
                at_border = y == self.size_y - 1
                Util.add_row_separator(display_map, ypos, cell_width, 
                        border=at_border)
                ypos += 1

            display_map = np.flip(display_map, axis=1)

            return display_map

        def print_display_map(display_map):
            xdim = display_map.shape[0]
            ydim = display_map.shape[1]

            for y in range(ydim):
                for x in range(xdim):
                    print(display_map[x,y], end='')
                print()

        # build cells
        cells = []
        for x in range(map_width):
            cell_column = []
            for y in range(map_height):
                cell_column.append(build_cell(x, y))
            cells.append(cell_column)
        cells = np.array(cells)

        # build display map
        display_map = build_map(cells)

        # print display map
        print_display_map(display_map)

    def located_borders(self, world_size):
        self.found_borders = True
        self._constrict_dims(world_size)

    def update(self, x, y, percept):
        # check if x and y are out of our current expected world size
        if x >= self.size_x or y >= self.size_y:
            self._expand_dims()

        # ensure current position is set visited and OK
        self.world_map[x, y, self.index_map["ok"]] = 1
        self.world_map[x, y, self.index_map["visited"]] = 1
        self.world_map[x, y, self.index_map["possible_wumpus"]] = 0
        self.world_map[x, y, self.index_map["possible_pit"]] = 0

        # handle screams
        if percept["scream"] is True:
            # TODO if we heard the scream and there is a possible wumpus in
            # front of us. It must be the wumpus location.
            self.seen_scream = True
            self.found_wumpus = True
            self._clear_wumpus()

        # handle glitter
        if percept["glitter"] is True:
            self.world_map[x, y, self.index_map["glitter"]] = 1
            self.gold_loc = (x, y)
        else:
            self.world_map[x, y, self.index_map["glitter"]] = 0

        # mark map with breezes and stenches before updating
        if percept["stench"] is True and self.seen_scream == False\
                and self.found_wumpus == False:
            self.world_map[x, y, self.index_map["stench"]] = 1
        if percept["breeze"] is True:
            self.world_map[x, y, self.index_map["breeze"]] = 1

        # handle stenches
        if percept["stench"] is True and self.seen_scream == False\
                and self.found_wumpus == False:
            self._update_neighbors(x, y, "stench")
        else:
            self._update_neighbors(x, y, "no_stench")

        # handle breezes
        if percept["breeze"] is True:
            self._update_neighbors(x, y, "breeze")
        else:
            self._update_neighbors(x, y, "no_breeze")

        # handle safety
        if (percept["stench"] is False or self.seen_scream == True) \
            and percept["breeze"] is False:
            self._update_neighbors(x, y, "ok")

        # double check possibilities
        if percept["stench"] is False or percept["breeze"] is False:
            # double check if any neighbors are now ok
            self._update_neighbors(x, y, "check_ok")

    def get_path(self, startx, starty, start_direction, destination=None):
        """ return a path to the nearest safe unvisited location
            if destination is specified then return shortest path to that
            destination

            will only use safe paths

            if no safe unvisited locations or no path to destination
                will return None
        """
        # TODO factor in turning time into the cost
        class Location:
            def __init__(self, x, y, direction, val, path):
                self.x = x
                self.y = y
                self.val = val
                self.path = path
                # direction when we entered this location
                self.direction = direction

            def __lt__(self, other):
                return self.val < other.val

            def __str__(self):
                return "({}, {}, {})".format(self.x, self.y, self.val)

        def get_turn_cost(loc0, loc1):
            """ return turn cost and resulting direction
            """
            def get_cost(desired_direction, current_direction):
                direction_map = {
                        "north" : 0,
                        "east" : 1,
                        "south" : 2,
                        "west" : 3                    
                        }
                
                desired_direction = direction_map[desired_direction]
                current_direction = direction_map[current_direction]

                if desired_direction == current_direction:
                    return 0

                # right turn
                rt_direction = current_direction
                lt_direction = current_direction
                distance = 0

                while True:
                    rt_direction += 1
                    lt_direction -= 1
                    rt_direction = rt_direction % 4
                    lt_direction = lt_direction % 4
                    distance += 1
                    
                    if rt_direction == desired_direction \
                        or lt_direction == desired_direction:
                        return distance

            pos0 = (loc0.x, loc0.y)
            pos1 = (loc1.x, loc1.y)
            desired_direction = Util.get_direction(pos0, pos1)

            cost = get_cost(desired_direction, loc0.direction)

            return cost, desired_direction

        # add first element to frontier
        start_loc = Location(startx, starty, start_direction, 0, 
                [(startx, starty)])

        frontier = [start_loc]
        touched = set()
    
        while len(frontier) > 0:
            # get current best node
            current = heapq.heappop(frontier)
            cx = current.x
            cy = current.y

            # check if this node has been touched before
            pos = (cx, cy)
            if pos in touched:
                continue
            else:
                touched.add(pos)

            # check if this node is a goal
            if destination is not None:
                if cx == destination[0] and cy == destination[1]:
                    return current.path
            else:
                if self.get(cx, cy, "ok") == 1 \
                and self.get(cx, cy, "visited") == 0:
                    return current.path

            # expand current node
            for nx, ny in self._get_neighbors(cx, cy):
                # check if neighbor is valid
                if self.get(nx, ny, "ok") == 1:
                    # add to frontier
                    new_path = copy.deepcopy(current.path)
                    new_path.append((nx, ny))
                    neighbor_loc = Location(nx, ny, None, 0, new_path)
                    # get cost of node
                    cost, neighbor_direction = get_turn_cost(current, neighbor_loc)
                    # plus one for the action of moving to the node
                    cost += 1
                    neighbor_loc.direction = neighbor_direction
                    neighbor_loc.val = current.val + cost

                    heapq.heappush(frontier, neighbor_loc)

        # couldn't find anything return None
        return None

    def get_pos(self, x, y):
        return self.world_map[x, y]

    def get_flat_map(self):
        return world_map.flatten()

    def found_gold(self):
        return self.gold_loc is not None

    def get_gold_loc(self):
        return self.gold_loc

    def get(self, x, y, index):
        return self.world_map[x, y, self.index_map[index]]

    def set(self, x, y, index, value):
        self.world_map[x, y, self.index_map[index]] = value

    def _constrict_dims(self, size):
        """ constrict the map to the given final size
        """
        
        new_map = np.zeros((size, size, self.vector_dim), dtype=np.int)
        new_map[:, :, :] = self.world_map[:size, :size, :]
    
        self.size_x = size
        self.size_y = size
        self.world_map = new_map

    def _expand_dims(self):
        new_size = self.size_x * 2

        new_map = np.zeros((new_size + 2, new_size + 2, self.vector_dim), 
                dtype=np.int)
    
        new_map[:self.size_x + 2, :self.size_y + 2, :] = self.world_map

        self.size_x = new_size
        self.size_y = new_size
        self.world_map = new_map

    def _clear_wumpus(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                self.set(x, y, "possible_wumpus", -1)
                self.set(x, y, "stench", 0)
                self.set(x, y, "wumpus", 0)

                # check if this makes any new locations ok
                if self.get(x, y, "breeze") == 0 \
                    and self.get(x, y, "stench") == 0 \
                    and self.get(x, y, "visited") == 1:
                    self._update_neighbors(x, y, "ok")

    def _clear_possible_wumpus(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                self.set(x, y, "possible_wumpus", 0)
                self.set(x, y, "stench", 0)

                # check if this makes any new locations ok
                if self.get(x, y, "breeze") == 0 \
                    and self.get(x, y, "stench") == 0 \
                    and self.get(x, y, "visited") == 1:
                    self._update_neighbors(x, y, "ok")

    def _update_neighbors(self, x, y, value):
        for pos in self._get_neighbors(x, y):
            cx = pos[0]
            cy = pos[1]

            if value == "stench":
                if self.get(cx, cy, "ok") != 1 \
                    and self.get(cx, cy, "possible_wumpus") != -1:
                    self.set(cx, cy, "possible_wumpus", 1)
                    self._check_found(cx, cy, "possible_wumpus")

            elif value == "breeze":
                if self.get(cx, cy, "ok") != 1 \
                    and self.get(cx, cy, "possible_pit") != -1:
                    self.set(cx, cy, "possible_pit", 1)

            elif value == "no_breeze":
                run_check = False
                if self.get(cx, cy, "possible_pit") == 1:
                    run_check = True
                self.set(cx, cy, "possible_pit", -1)
                if run_check == True:
                    self._update_neighbors(cx, cy, "check_pit")

            elif value == "no_stench":
                run_check = False
                if self.get(cx, cy, "possible_wumpus") == 1:
                    run_check = True
                self.set(cx, cy, "possible_wumpus", -1)
                if run_check == True:
                    self._update_neighbors(cx, cy, "check_wumpus")

            elif value == "ok":
                if self.get(cx, cy, "wumpus") != 1 \
                    and self.get(cx, cy, "pit") != 1:
                    self.set(cx, cy, "ok", 1)

            elif value == "check_pit":
                self._check_found(cx, cy, "breeze")

            elif value == "check_wumpus":
                self._check_found(cx, cy, "stench")

            elif value == "check_ok":
                if self.get(cx, cy, "possible_wumpus") == -1 \
                    and self.get(cx, cy, "possible_pit") == -1 \
                    and self.get(cx, cy, "pit") != 1 \
                    and self.get(cx, cy, "wumpus") != 1:
                    self.set(cx, cy, "ok", 1)

        # handle finding a pit or wumpus
        self._check_found(x, y, value)

    def _check_found(self, x, y, value):
        if value != "stench" and value != "breeze" \
            and value != "possible_wumpus":
            return

        if self.get(x, y, value) == 1:
            if value == "stench" and self.found_wumpus == False:
                # check if we know wumpus is neighboring
                possible_wumpi = self._find_neighboring(x, y, "possible_wumpus")
                if len(possible_wumpi) == 1:
                    self.found_wumpus = True
                    wumpus_pos = possible_wumpi[0]
                    self.wumpus_loc = wumpus_pos
                    self.set(wumpus_pos[0], wumpus_pos[1], "wumpus", 1)
                    self._clear_possible_wumpus()

            elif value == "breeze":
                # find neighboring pp and mark it pit
                possible_pits = self._find_neighboring(x, y, "possible_pit")
                actual_pits = self._find_neighboring(x, y, "pit")
                if len(possible_pits) == 1 and len(actual_pits) == 0:
                    pit_pos = possible_pits[0]
                    self.set(pit_pos[0], pit_pos[1], "pit", 1)
                    self.set(pit_pos[0], pit_pos[1], "possible_pit", 0)

            elif value == "possible_wumpus" and self.found_wumpus == False:
                stenches = self._find_neighboring(x, y, "stench")
                # if more than one stench surrounding we know this is the wumpus
                # TODO bug when there are two possible wumpi and two stenches
                # in a 2x2 area this code will mark the wrong one 
                # 50% of the time
                if len(stenches) > 1:
                    # mark that we found the wumpus
                    self.found_wumpus = True
                    self.wumpus_loc = (x, y)
                    self.set(x, y, "wumpus", 1)
                    self._clear_possible_wumpus()

    def _find_neighboring(self, x, y, index):
        neighboring = []
        for (cx, cy) in self._get_neighbors(x, y):
            if self.get(cx, cy, index) == 1:
                neighboring.append((cx, cy))
        return neighboring

    def _get_neighbors(self, x, y):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        # not affected by border size
        if x > 0:
            min_x = -1
        if y > 0:
            min_y = -1

        # is affected by border size
        if x < self.size_x - 1 or self.found_borders is False:
            max_x = 1
        if y < self.size_y - 1 or self.found_borders is False:
            max_y = 1

        for nx in range(x + min_x, x + max_x + 1):
            if nx != x:
                yield (nx, y)
        for ny in range(y + min_y, y + max_y + 1):
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
        self.world_map = Map()

    def destructor(self):
        pass

    def initialize(self):
        self.last_action = None
        self.x = 0
        self.y = 0
        self.direction = "east"
        self.hasgold = False
        self.hasarrow = True
        self.path = []
        self.leave = False

        self.map.reset()

    def process(self, stench, breeze, glitter, bump, scream):
        """ process the percept and return desired action

        input:
            stench - boolean, whether stench is detected
            breeze - boolean, whether breeze is detected
            glitter - boolean, whether glitter is detected
            bump - boolean, whether bump has resulted from the last action
            scream - boolean, whether scream has resulted from the last action
        """
        # pack up the percept
        percept = {
            "stench" : stench,
            "breeze" : breeze,
            "glitter" : glitter,
            "bump" : bump,
            "scream" : scream
        }

        # TODO add logic for shooting the wumpus

        # update location
        if self.last_action == Action.GOFORWARD and bump is False:
            self.update_location()

        # update map
        if bump is True:
            # found the border of the world. report to map.
            self.world_map.located_borders(max(self.x, self.y) + 1)
        self.world_map.update(self.x, self.y, percept)

        current_action = None

        # set up a path if we need one
        if len(self.path) <= 1:
            # if have gold then go to start
            if self.hasgold is True:
                self.path = self.world_map.get_path(self.x, self.y, self.direction, (0, 0))
            
            # if map has found gold then go there
            elif (self.world_map.found_gold() is True) and self.hasgold is False:
                # TODO need to make sure wumpus isn't blocking the path
                # for repeated attempts
                gold_loc = self.world_map.get_gold_loc()
                self.path = self.world_map.get_path(self.x, self.y, self.direction, gold_loc)

            # if nothing else then go to nearest safe place
            else:
                self.path = self.world_map.get_path(self.x, self.y, self.direction)

                if self.path is None:
                    # no more reachable safe places
                    # try to leave
                    self.path = self.world_map.get_path(self.x, self.y, self.direction, (0, 0))
                    self.leave = True

        # if glitter then grab gold
        if glitter is True:
            self.hasgold = True
            current_action = Action.GRAB
            self.path = self.world_map.get_path(self.x, self.y, self.direction, (0, 0))

        #  if can win then win
        elif self.x == 0 and \
            self.y == 0 and \
            self.hasgold is True and \
            current_action is None:
            current_action = Action.CLIMB

        # if we want to leave then leave
        if self.x == 0 and \
            self.y == 0 and \
            self.leave is True and \
            current_action is None:
            current_action = Action.CLIMB


        # if have path to follow then follow path
        if len(self.path) > 1 and current_action is None:
            current_action = self.follow_path(self.path)

        # print out stuff
        print("position: {}, {}".format(self.x, self.y))
        print("direction: {}".format(self.direction))
        print("current path: {}".format(self.path))
        print()
        self.world_map.print()
        print()
        print(end='', flush=True)

        # if turn action then update orientation
        if current_action == Action.TURNLEFT \
            or current_action == Action.TURNRIGHT:
                self.update_orientation(current_action)

        assert current_action is not None

        self.last_action = current_action
        return current_action

    def gameover(self):
        pass

    def follow_path(self, path):
        """ get the action to take to follow the given path
            maintain the path along the way.

            first node in path should be your current location
        """
        assert path[0][0] == self.x and path[0][1] == self.y

        def get_turn_direction(desired_direction, current_direction):
            # TODO a little inelegant/ineffecient
            direction_map = {
                    "north" : 0,
                    "east" : 1,
                    "south" : 2,
                    "west" : 3                    
                    }
            
            desired_direction = direction_map[desired_direction]
            current_direction = direction_map[current_direction]

            # right turn
            rt_direction = current_direction
            lt_direction = current_direction
            while True:
                rt_direction += 1
                lt_direction -= 1
                rt_direction = rt_direction % 4
                lt_direction = lt_direction % 4
                
                if rt_direction == desired_direction:
                    return Action.TURNRIGHT
                elif lt_direction == desired_direction:
                    return Action.TURNLEFT
        
        desired_direction = Util.get_direction(path[0], path[1])

        if desired_direction == self.direction:
            path.pop(0)
            return Action.GOFORWARD
        else:
            return get_turn_direction(desired_direction, self.direction)

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
