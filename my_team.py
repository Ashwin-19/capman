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
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point, Queue
from team_utils import astar_search

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveCapMan', second='DefensiveCapMan', num_training=0):
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

class OffensiveCapMan(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        # start in offense
        self.attack = True
        self.last_state_attack = True

        # track food collected
        self.food_collected = 0
        self.food_last_turn = 0

        # track power usage and if pacman is scared
        self.power = False
        self.power_turns = 40
        self.capsules_last_turn = 1
        self.scared_timer = 0

        # track recent actions
        self.cache = []
        self.cache_use = 1

        # track moves
        self.moves_left = 300

        # track safe points
        self.safe_points = []
        self.boundary = None
        self.caution = False

        # track goals
        self.last_goal = None


    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)

        self.start = game_state.get_agent_position(self.index)
        self.food_last_turn = set(self.get_food(game_state).as_list())
        self.capsules_last_turn = set(self.get_capsules(game_state))

        if self.capsules_last_turn:
            self.last_goal = min(
                self.capsules_last_turn,
                key=lambda p: self.get_maze_distance(self.start,p)
            )
        elif self.food_last_turn:
            self.last_goal = min(
                self.food_last_turn,
                key=lambda p: self.get_maze_distance(self.start,p)
            )
        else:
            self.last_goal = self.start

        self.boundary = game_state.data.layout.width//2 -1*(self.red) + 1*(1-self.red)
        for y_safe in range(game_state.data.layout.height):
            if not game_state.data.layout.walls[self.boundary][y_safe]:
                self.safe_points.append((self.boundary,y_safe))


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # track food, capsules and moves left
        food_left = set(self.get_food(game_state).as_list())
        capsules_left = set(self.get_capsules(game_state))
        self.update_food_count(game_state,food_left)
        self.update_capsules(capsules_left)
        self.scared_timer = game_state.get_agent_state(self.index).scared_timer
        self.food_last_turn = food_left
        self.capsules_last_turn = capsules_left
        self.moves_left -= 1
        self.caution = False

        # track recent states to avoid LOOPS
        pacman = game_state.get_agent_position(self.index)
        if len(self.cache) < 7:
            self.cache.append(pacman)
        else:
            self.cache = self.cache[1:] + [pacman]
        # Reset cache use each turn
        self.cache_use += 1

        # Avoid STOPPING
        actions = game_state.get_legal_actions(self.index)
        actions = [action for action in actions if action != Directions.STOP]
        enemy, enemy_distance = self.get_closest_enemy(game_state)
        _, enemy_state = enemy
        enemy_xy = enemy_state.get_position()
        safe_point, safe_distance = self.get_closest_safe_point(game_state)

        if len(food_left) <= 2:
            self.attack = False
        elif self.in_home_territory(game_state):
            self.attack=True
            if any((
                enemy_xy is not None and pacman[0]==safe_point[0] and\
                ((self.red and enemy_xy[0]>pacman[0]) or (not self.red and enemy_xy[0]<pacman[0])),
                self.scared_timer > enemy_distance and enemy_distance <= 2
            )):
                safe_actions = self.get_safe_actions(game_state,actions)
                actions = safe_actions if safe_actions else actions
                self.caution = True
                # suspend cache for 5 moves for safety
                self.cache_use = -4 if self.cache_use else self.cache_use
        elif self.moves_left <= safe_distance:
            self.attack = False
        elif self.power:
            self.attack = enemy_state.scared_timer>4 or enemy_distance>3
            self.power_turns -= 1
        elif enemy_distance <= 4:
            self.attack = False
            if enemy_distance <= 2:
                self.cache_use = -4 if self.cache_use else self.cache_use
                self.caution = True
        else:
            self.attack = True

        # NO LOOPS unless absolutely necessary
        if self.cache_use and len(actions)>1:
            actions = list(filter(
                lambda action: self.get_successor(game_state,action)\
                    .get_agent_position(self.index) not in self.cache[-(len(actions)+1):],
                actions
            ))

        if self.attack:

            if not self.last_state_attack:
                self.last_goal = self.get_next_goal(game_state,randomized=True)
            elif self.last_goal==pacman:
                self.last_goal = self.get_next_goal(game_state)

            if self.last_goal is not None:
                best_action = astar_search(
                    agent=self,
                    game_state=game_state,
                    goal=self.last_goal,
                    mode="offense_a"
                )
                if best_action not in actions and self.caution:
                    best_action = random.choice(actions)
            else:
                best_action = random.choice(actions)

        else:
            goal = safe_point
            best_action = astar_search(
                agent = self,
                game_state = game_state,
                goal = goal,
                mode = "offense_f"
            )
            # Do not get trapped by opponent in a loop
            if self.cache_use and len(set(self.cache))==2:
                return random.choice(actions)

        self.last_state_attack = self.attack

        legal_actions = game_state.get_legal_actions(self.index)
        if best_action is None or best_action not in legal_actions:
            return random.choice(game_state.get_legal_actions(self.index))

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

    def update_food_count(self,game_state,food_left):
        difference = self.food_last_turn - food_left
        if difference:
            self.food_collected += 1
        if self.in_home_territory(game_state):
            self.food_collected = 0

    def update_capsules(self,capsules_left):
        difference = self.capsules_last_turn - capsules_left
        if difference:
            self.power = True
            self.power_turns = 40
        if not self.power_turns:
            self.power = False
            self.power_turns = 0


    def in_home_territory(self,game_state):
        x,_ = game_state.get_agent_position(self.index)
        if self.red:
            return x <= self.boundary
        return x >= self.boundary

    def get_closest_safe_point(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        safe_point = random.choice(self.safe_points)
        distance = float("inf")
        for point in self.safe_points:
            if self.get_maze_distance(pacman,point) < distance:
                distance = self.get_maze_distance(pacman,point)
                safe_point = point
        return safe_point, distance


    def get_closest_enemy(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        distance = 6
        enemies = self.get_opponents(game_state)
        closest = random.choice(enemies)
        closest = (closest, game_state.get_agent_state(closest))
        for enemy in enemies:
            enemy_state = game_state.get_agent_state(enemy)
            if not enemy_state.is_pacman and enemy_state.get_position() is not None:
                if self.get_maze_distance(pacman,enemy_state.get_position()) < distance:
                    distance = self.get_maze_distance(pacman,enemy_state.get_position())
                    closest = (enemy, enemy_state)
        return closest, distance


    def get_safe_actions(self,game_state,actions):
        next_states = list(map(
            lambda action: (
                action,
                self.get_successor(game_state,action).get_agent_state(self.index).get_position()
            ),
            actions
        ))
        enemy_states = set()
        for enemy in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(enemy)
            if not enemy_state.is_pacman and enemy_state.get_position() is not None:
                enemy_actions = game_state.get_legal_actions(enemy)
                for action in enemy_actions:
                    enemy_states.add(
                        game_state.generate_successor(enemy,action).\
                            get_agent_state(enemy).get_position()
                    )
        safe_actions = [state[0] for state in next_states if state[1] not in enemy_states]
        return safe_actions


    def get_next_goal(self,game_state,randomized=False):
        pacman = game_state.get_agent_position(self.index)
        all_goals = self.get_capsules(game_state) + self.get_food(game_state).as_list()
        if not all_goals:
            return None
        if randomized:
            midpoint = game_state.data.layout.height//2
            filtered = list(filter(lambda goal: goal[1] >= midpoint, all_goals)) if pacman[1] < midpoint\
                else list(filter(lambda goal: goal[1] <= midpoint, all_goals))
            all_goals = filtered if filtered else all_goals
            top_k = sorted(all_goals, key=lambda goal: self.get_maze_distance(pacman,goal))[:5]
            return random.choice(top_k)
        return min(all_goals, key=lambda goal: self.get_maze_distance(pacman,goal))


class DefensiveCapMan(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        # start in patrol
        self.patrol = True
        self.last_state_patrol = True
        self.last_goal = self.start

        # track food left in each turn
        self.food_lost = 0
        self.food_last_turn = 0

        # track capsule usage and if pacman is scared
        self.is_scared = False
        self.scared_timer = 40
        self.capsules_last_turn = 1

        # track moves
        self.moves_left = 300

        # track safe points, boundary and patrol points
        self.safe_points = []
        self.patrol_points = Queue()
        self.boundary = 0


    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.food_last_turn = set(self.get_food_you_are_defending(game_state).as_list())
        self.capsules_last_turn = set(self.get_capsules_you_are_defending(game_state))
        if self.capsules_last_turn:
            self.last_goal = max(
                self.capsules_last_turn,
                key=lambda p: self.get_maze_distance(self.start,p)
            )
        elif self.food_last_turn:
            self.last_goal = max(
                self.food_last_turn,
                key=lambda p: self.get_maze_distance(self.start,p)
            )
        else:
            self.last_goal = self.start
        self.boundary = game_state.data.layout.width//2 -1*(self.red) + 1*(1-self.red)
        for y_safe in range(game_state.data.layout.height):
            if not game_state.data.layout.walls[self.boundary][y_safe]:
                self.safe_points.append((self.boundary,y_safe))
        self.patrol_points.push(max(self.safe_points,key=lambda xy: xy[1]))
        self.patrol_points.push(min(self.safe_points,key=lambda xy: xy[1]))


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # track food, capsules and moves left
        food_left = set(self.get_food_you_are_defending(game_state).as_list())
        capsules_left = set(self.get_capsules_you_are_defending(game_state))
        food_lost = self.update_food_count(food_left)
        capsule_lost = self.update_capsules(capsules_left)
        self.scared_timer = game_state.get_agent_state(self.index).scared_timer
        self.food_last_turn = food_left
        self.capsules_last_turn = capsules_left
        self.moves_left -= 1

        pacman = game_state.get_agent_position(self.index)
        enemy, d_enemy, _ = self.get_enemy_info(game_state)
        enemy, enemy_state = enemy
        enemy_xy = enemy_state.get_position()

        if enemy_xy is not None and self.in_home_territory(game_state,agent=enemy) and\
        ((self.is_scared and self.scared_timer < d_enemy) or (not self.is_scared)):
            goal = enemy_xy
            self.patrol = False
        elif enemy_xy is not None and self.in_home_territory(game_state,agent=enemy) and\
        (self.is_scared and self.scared_timer > d_enemy):
            goal = max(
                self.safe_points,
                key=lambda point: self.get_maze_distance(enemy_xy,point)\
                    if (point != enemy_xy) and (point != pacman) else float("-inf")
            )
            self.patrol = False
        elif food_lost:
            goal = food_lost.pop()
            self.patrol = False
        elif capsule_lost:
            goal = capsule_lost.pop()
            self.patrol = False
        elif self.last_goal != self.start and self.last_goal != pacman:
            goal = self.last_goal
        else:
            goal = self.patrol_points.pop()
            self.patrol_points.push(goal)
            self.patrol = True
            if goal==pacman:
                goal = self.patrol_points.pop()
                self.patrol_points.push(goal)

        best_action = astar_search(
            agent = self,
            game_state = game_state,
            goal = goal,
            mode = "defense"
        )

        self.last_state_patrol = self.patrol
        self.last_goal = goal

        legal_actions = game_state.get_legal_actions(self.index)
        if best_action is None or best_action not in legal_actions:
            return random.choice(game_state.get_legal_actions(self.index))

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


    def get_enemy_info(self,game_state):
        pacman = game_state.get_agent_position(self.index)
        n_invaders = 0
        distance = 6
        enemies = self.get_opponents(game_state)
        closest = random.choice(enemies)
        closest = (closest, game_state.get_agent_state(closest))
        for enemy in enemies:
            enemy_state = game_state.get_agent_state(enemy)
            if enemy_state.is_pacman and enemy_state.get_position() is not None:
                if self.get_maze_distance(pacman,enemy_state.get_position()) < distance:
                    distance = self.get_maze_distance(pacman,enemy_state.get_position())
                    closest = (enemy, enemy_state)
                n_invaders += 1
        return closest, distance, n_invaders


    def update_food_count(self,food_left):
        difference = self.food_last_turn - food_left
        if difference:
            self.food_lost += 1
        else:
            self.food_lost = 0
        return difference


    def update_capsules(self,capsules_left):
        difference = self.capsules_last_turn - capsules_left
        if difference:
            self.scared_timer = 40
            self.is_scared = True
        if not self.scared_timer:
            self.is_scared = False
        return difference


    def in_home_territory(self,game_state,agent=None):
        if agent is None:
            x,_ = game_state.get_agent_position(self.index)
        else:
            x,_ = game_state.get_agent_state(agent).get_position()
        if self.red:
            return x <= self.boundary
        return x >= self.boundary
