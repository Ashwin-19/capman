from contest.util import PriorityQueue, Queue
from capture import AgentRules
from game import Directions, Actions
from copy import deepcopy


def get_legal_actions(agent, game_state, pos):
    agent_state = game_state.get_agent_state(agent.index)
    config = deepcopy(agent_state.configuration)
    config.pos = pos
    legal_actions = AgentRules.filter_for_allowed_actions(
        Actions.get_possible_actions(
            config=config,
            walls=game_state.data.layout.walls
        )
    )
    return list(filter(
        lambda action: action!=Directions.STOP,
        legal_actions
    ))


def get_distance_from_enemy(agent,game_state,pos,mode):

    distance = 100
    enemies = agent.get_opponents(game_state)

    for enemy in enemies:

        enemy_state = game_state.get_agent_state(enemy)

        if mode=="offense" and not enemy_state.is_pacman and enemy_state.get_position() is not None:
            distance = min(distance,agent.get_maze_distance(pos,enemy_state.get_position()))

        if mode=="defense" and enemy_state.is_pacman and enemy_state.get_position() is not None:
            distance = min(distance,agent.get_maze_distance(pos,enemy_state.get_position()))

    return distance


def get_distance_from_capsule(agent,game_state,pos):
    capsules = agent.get_capsules(game_state)
    if capsules:
        return min(list(map(
            lambda capsule: agent.get_maze_distance(pos,capsule),
            capsules
        )))
    return -1


def get_distance_from_food(agent,game_state,pos):
    food_left = agent.get_food(game_state).as_list()
    if food_left:
        return min(list(map(
            lambda food: agent.get_maze_distance(pos,food),
            food_left
        )))
    return -1


def get_child(node,action):
    x, y = node
    dx, dy = Actions.direction_to_vector(action)
    return int(x + dx), int(y + dy)


def heuristic_offense(agent,game_state,pos,goal,mode):

    d_enemy = get_distance_from_enemy(agent,game_state,pos,mode="offense")
    d_power = get_distance_from_capsule(agent,game_state,pos)
    d_food = get_distance_from_food(agent,game_state,pos)
    d_goal = agent.get_maze_distance(pos,goal)

    h = d_goal

    if mode=="flee":
        if d_enemy==0 and not agent.power:
            return "blocked"
        if d_power==0:
            h -= 6
        if d_food==0:
            h -= 1.5
        if d_enemy < 100 and not agent.power:
            h += (10*(100-d_enemy))
    else:
        if not agent.power and d_enemy <= 100 and not in_home_territory(agent,pos):
            h += 12*(100-d_enemy)
    return h


def heuristic_defense(agent,game_state,pos,goal):

    d_enemy = get_distance_from_enemy(
        agent, game_state, pos, mode="defense"
    )
    h = agent.get_maze_distance(pos,goal)

    if agent.is_scared and d_enemy==0:
        return "blocked"

    if not agent.is_scared and d_enemy < 100:
        h -= 10*(100-d_enemy)

    return h


def in_home_territory(agent,child):
    if agent.red:
        return child[0]<=agent.boundary
    return child[0]>=agent.boundary


def astar_search(agent,game_state,goal,mode):
    """Search the node that has the lowest combined cost and heuristic first."""
    # get start state and initial value for heuristic
    start = game_state.get_agent_position(agent.index)

    if start==goal:
        if mode in ("offense_f","offense_a"):
            return max(
                get_legal_actions(agent, game_state, pos=start),
                key=lambda action: heuristic_offense(
                    agent,game_state,get_child(start,action),goal,mode=mode
                )
            )
        return max(
            get_legal_actions(agent, game_state, pos=start),
            key=lambda action: heuristic_defense(
                agent,game_state,get_child(start,action),goal
            )
        )

    # initialize dict for expanded nodes and store lowest costs to reach them
    expanded = {}

    # frontier for nodes that have been generated, but whose children have not been computed
    frontier = PriorityQueue()

    # add (start state, actions), heuristic to frontier
    # we do not compute heursitic for the start state, as it will pop first independently
    frontier.push((start,Queue(),0), 0)

    while not frontier.is_empty():

        # fetch the node, and path upto node with the lowest cost
        node, actions, cost_upto_node = frontier.pop()

        # if the node is goal, then return the next action
        if node==goal:
            return actions.pop()

        # only visit the node if it has not been visited before or if
        # a path with less associated cost has been discovered.
        if (node not in expanded) or (expanded[node] > cost_upto_node):

            # update the associated cost with the node
            expanded[node] = cost_upto_node

            # add each child to frontier, while continuing to track the path
            # and the cost associated with the actions to reach it
            for action in get_legal_actions(agent, game_state, pos=node):
                child = get_child(node,action)
                cost_child = 1
                if mode in ("offense_f","offense_a"):
                    h = heuristic_offense(agent,game_state,child,goal,mode=mode)
                    if h!="blocked":
                        actionq = deepcopy(actions)
                        actionq.push(action)
                        frontier.update(
                            (child, actionq, cost_upto_node + cost_child),
                            cost_upto_node + cost_child + h
                        )
                else:
                    if in_home_territory(agent,child):
                        h = heuristic_defense(agent,game_state,child,goal)
                        if h!="blocked":
                            actionq = deepcopy(actions)
                            actionq.push(action)
                            frontier.update(
                                    (child, actionq, cost_upto_node + cost_child),
                                    cost_upto_node + cost_child + h
                                )
