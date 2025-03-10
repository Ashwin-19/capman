# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions, Actions
from util import nearest_point, flip_coin


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='QLearningCapMan', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class QLearningCapMan(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.eps = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        self.weights_attack = {'bias': -0.8233704631746798, 'd_food': 11.40655347843293, 'd_opp': -0.2994842432459648, 'score': 20.0}
        self.weights_flee = {'bias': -1.2041591279750847, 'd_near': 9.764211641272448, 'd_opp': 300, 'score': 20.0}
        self.mode = "attack" # or can be flee
        self.food_collected = 0
        self.food_last_turn = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.food_last_turn = len(self.get_food(game_state).as_list())
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
        self.update_food_count(game_state,food_left)
        self.food_last_turn = food_left

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        if any((
            not self.in_home_territory(game_state) and self.get_distance_from_opponent(game_state)<=5,
            self.food_collected >= 3
        )):
            self.mode = "flee"
        else:
            self.mode = "attack"

        if self.mode=="attack" and flip_coin(self.eps):
            best_actions = [random.choice(actions)]
        else:
            values = [self.evaluate(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]

        best_action = random.choice(best_actions)
        self.update_q(game_state,best_action)
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        if self.food_collected:
            x,y = game_state.get_agent_position(self.index)
            x_safe = game_state.data.layout.width//2 + 1*(1-self.red)
            if x==x_safe+1 and not game_state.data.layout.walls[x-1][y]:
                best_action = Directions.WEST
            if x==x_safe-1 and not game_state.data.layout.walls[x+1][y]:
                best_action = Directions.EAST

        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        q = self.qlookup(game_state,action)
        return q

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        successor = self.get_successor(game_state, action)
        pacman = successor.get_agent_position(self.index)
        features = util.Counter()
        features["bias"] = 1
        food_distances = [self.get_maze_distance(pacman, food) for food in self.get_food(game_state).as_list()]

        if self.mode == "attack":
            features["d_food"] = 1/(min(food_distances)+1)
        else:
            safe_point = self.get_closest_safe_point(game_state)
            features["d_near"] = 1/(self.get_maze_distance(pacman,safe_point)+1)

        features["d_opp"] = self.get_distance_from_opponent(successor)
        features["score"] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return self.weights

    def qlookup(self,game_state,action):
        features = self.get_features(game_state,action)
        if self.mode=="attack":
            return sum([self.weights_attack.get(f,0)*v for f,v in features.items()])
        return sum([self.weights_flee.get(f,0)*v for f,v in features.items()])

    def get_reward(self,game_state,action):

        reward = 0
        successor = self.get_successor(game_state, action)
        x,y = successor.get_agent_position(self.index)

        if self.mode=="flee":
            d_prev = self.get_distance_from_opponent(game_state)
            d_next = self.get_distance_from_opponent(successor)
            if d_next < d_prev:
                reward -= 2*(d_prev-d_next)
            else:
                reward += 2*(d_next-d_prev)

        if self.get_score(successor) > self.get_score(game_state):
            reward += 5 * (self.get_score(successor) - self.get_score(game_state))

        if (x,y) == self.start:
            reward -= 10

        if self.mode == "attack":

            if self.get_food(game_state)[x][y]:
                reward += 2

            d_prev = self.get_distance_to_closest_food(game_state)
            d_next = self.get_distance_to_closest_food(successor)
            if d_next < d_prev:
                reward += (d_prev-d_next)
            else:
                reward -= (d_next-d_prev)

        return reward

    def update_q(self,game_state,action):
        features = self.get_features(game_state,action)
        q_curr = self.qlookup(game_state,action)
        reward = self.get_reward(game_state,action)
        actions = game_state.get_legal_actions(self.index)
        max_next_q = max([self.evaluate(game_state, a) for a in actions])
        target = reward + self.gamma * max_next_q
        error = target - q_curr
        for f,v in features.items():
            if self.mode == "attack":
                self.weights_attack[f] = self.weights_attack.get(f,0) + self.alpha * error * v
            else:
                self.weights_flee[f] = self.weights_flee.get(f,0) + self.alpha * error * v

    def update_food_count(self,game_state,food_left):
        if food_left < self.food_last_turn:
            self.food_collected += 1
        if any((
            game_state.get_agent_position(self.index) == self.start,
            self.in_home_territory(game_state)
        )):
            self.food_collected = 0

    def in_home_territory(self,game_state):
        x,_ = game_state.get_agent_position(self.index)
        if self.red:
            return x <= game_state.data.layout.width//2
        return x > game_state.data.layout.width//2

    def get_closest_safe_point(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        safe_points = self.get_all_safe_points(game_state)
        safe_point = safe_points[0]
        d = 1e3
        for point in safe_points:
            if self.get_maze_distance(pacman,point) < d:
                d = self.get_maze_distance(pacman,point)
                safe_point = point
        return safe_point

    def get_all_safe_points(self,game_state):
        x_safe = game_state.data.layout.width//2 + 1*(1-self.red)
        safe_points = []
        for y_safe in range(game_state.data.layout.height):
            if not game_state.data.layout.walls[x_safe][y_safe]:
                safe_points.append((x_safe,y_safe))
        return safe_points

    def get_distance_from_opponent(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        distance = 6
        for opp in self.get_opponents(game_state):
            opponent = game_state.get_agent_state(opp)
            if not opponent.is_pacman and opponent.get_position() is not None:
                distance = min(
                    distance,
                    self.get_maze_distance(
                        pacman,
                        opponent.get_position()
                    )
                )
        return distance

    def get_closest_opponent(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        distance = 6
        closest = None
        for opp in self.get_opponents(game_state):
            opponent = game_state.get_agent_state(opp)
            if not opponent.is_pacman and opponent.get_position() is not None:
                if self.get_maze_distance(pacman,opponent.get_position()) < distance:
                    distance = self.get_maze_distance(pacman,opponent.get_position())
                    closest = opponent.get_position()
        return closest
    
    def get_distance_to_closest_food(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        return min([self.get_maze_distance(pacman, food) for food in self.get_food(game_state).as_list()])
    

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
