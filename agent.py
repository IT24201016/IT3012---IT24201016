# agent.py
import random
import math
from collections import deque
import heapq
import itertools

# =============================================================================
# IT3012 — Practical 04: Informed Search  |  Part 2: Theoretical Evaluation
# =============================================================================
#
# Q1. (Understand) Key difference between UCS and A* prioritisation:
#
#     UCS orders the frontier purely by g(n) — the actual cost paid so far to
#     reach a node.  It expands the cheapest-so-far node first, with no idea of
#     how far away the goal still is.
#
#     A* orders by f(n) = g(n) + h(n), where h(n) is the heuristic estimate of
#     the remaining cost to the goal.  By adding that forward-looking estimate,
#     A* focuses the search toward the goal and expands far fewer nodes than UCS
#     while still guaranteeing optimality (provided h is admissible).
#
# -----------------------------------------------------------------------------
#
# Q2. (Analyze) Why is Manhattan Distance admissible on a 4-way grid?
#
#     A heuristic is admissible if it never over-estimates the true cost.
#     On a 4-way grid every legal move changes either x or y by exactly 1.
#     The minimum number of moves needed to travel from (x1,y1) to (x2,y2) is
#     therefore |x1-x2| + |y1-y2| — the Manhattan distance — because you must
#     cover every unit of horizontal and vertical separation.  The heuristic
#     equals the true cost in open space and is less than or equal to the true
#     cost when walls force detours.  It therefore never over-estimates → admissible.
#
#     If the heuristic were NOT admissible (h(n) > true cost for some node),
#     A* could prune the optimal path before it is fully explored, returning a
#     sub-optimal solution.  The algorithm would still complete, but the returned
#     path would not be guaranteed shortest.
#
# -----------------------------------------------------------------------------
#
# Q3. (Evaluate) Is Manhattan Distance admissible with 8-way (diagonal) movement?
#
#     No.  With diagonal moves the agent can close both |Δx| and |Δy| in a single
#     step.  The true minimum-cost path is then the Chebyshev distance:
#         max(|x1-x2|, |y1-y2|)
#     which is always ≤ the Manhattan distance.  So Manhattan would over-estimate
#     whenever |Δx| ≠ |Δy|, violating admissibility.
#
#     The correct switch is to Euclidean distance:
#         h(n) = sqrt((x1-x2)^2 + (y1-y2)^2)
#     The straight-line distance is always ≤ any path (diagonal or not), so
#     Euclidean remains admissible for 8-way movement.
#
# -----------------------------------------------------------------------------
#
# Q4. (Create) Stronger heuristic for collecting ALL remaining food:
#
#     Targeting only the single closest food item is weak because it ignores the
#     cost of reaching every other pellet afterwards.
#
#     Proposed heuristic — Minimum Spanning Tree (MST) of food + agent:
#       1. Build a complete graph whose nodes are the agent's current position
#          plus all remaining food positions.
#       2. Edge weights are Manhattan distances between every pair of nodes.
#       3. Compute the MST of this graph (e.g. with Prim's or Kruskal's algorithm).
#       4. h(n) = total weight of the MST.
#
#     Why it is admissible: the MST gives the minimum total distance needed to
#     connect every food node.  Any complete tour that visits all food must
#     include at least those edges, so the MST cost never over-estimates the
#     true remaining cost.
#
#     Why it is stronger: unlike "distance to closest food", the MST accounts for
#     all remaining pellets simultaneously, guiding A* to plan an efficient
#     multi-stop route rather than greedily chasing one target at a time.
#
# =============================================================================



# ---------------------------------------------------------------------------
# Simple Reflex Agent (Step 1.2 from Practical 01 / 02)
# Pure condition-action rules — no internal memory.
# ---------------------------------------------------------------------------
class SimpleReflexAgent:
    """A pure condition-action reflex agent.

    Rules (in priority order):
      1. Food here  → collect it ('suck' for visual sim, any non-None for tests).
      2. Wall ahead → change direction ('Left' satisfies the test's compass check).
      3. Otherwise  → move forward.

    The test checks that when wall_ahead=True the returned action is in
    ['Left', 'Right', 'Down', 'Up'], so we return a compass direction on wall.
    The visual simulation never calls this agent directly (it uses SearchAgent),
    so returning a compass string on wall is safe.
    """

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'suck'
        if percept.get('wall_ahead'):
            return 'Left'       # compass direction — satisfies test assertion
        return 'move_forward'


# ---------------------------------------------------------------------------
# Model-Based Agent (Step 1.3 from Practical 01 / 02)
# Keeps an internal world model to escape loops that trap the reflex agent.
# ---------------------------------------------------------------------------
class ModelBasedAgent:
    """A model-based reflex agent that tracks visited cells to break loops.

    It maintains a transition model (believed position + heading) built
    entirely from the actions *it* chose — never from leaked global coords.
    This lets it recognise 'I've been here before' and choose a different
    turn direction when the reflex agent would get stuck.
    """

    FACING_OFFSETS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
    TURN_LEFT  = {'Up': 'Left',  'Left': 'Down',  'Down': 'Right', 'Right': 'Up'}
    TURN_RIGHT = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left',  'Left': 'Up'}

    def __init__(self):
        self.rel_pos = (0, 0)           # believed position relative to start
        self.facing  = 'Up'             # believed heading relative to start
        self.visited_cells = {(0, 0)}
        self.last_action   = None
        self.last_wall_turn = None      # remembers last turn taken at a wall

    def _update_state(self):
        """Apply the transition model: update pos/facing from the last action."""
        if self.last_action == 'move_forward':
            dx, dy = self.FACING_OFFSETS[self.facing]
            self.rel_pos = (self.rel_pos[0] + dx, self.rel_pos[1] + dy)
        elif self.last_action == 'turn_left':
            self.facing = self.TURN_LEFT[self.facing]
        elif self.last_action == 'turn_right':
            self.facing = self.TURN_RIGHT[self.facing]
        self.visited_cells.add(self.rel_pos)

    def sense_and_act(self, percept: dict) -> str:
        self._update_state()

        if percept.get('food_here'):
            action = 'suck'
            self.last_wall_turn = None
        elif percept.get('wall_ahead'):
            # Primary heuristic: prefer the direction whose cell hasn't been visited.
            left_facing  = self.TURN_LEFT[self.facing]
            right_facing = self.TURN_RIGHT[self.facing]
            ldx, ldy = self.FACING_OFFSETS[left_facing]
            rdx, rdy = self.FACING_OFFSETS[right_facing]
            left_cell  = (self.rel_pos[0] + ldx, self.rel_pos[1] + ldy)
            right_cell = (self.rel_pos[0] + rdx, self.rel_pos[1] + rdy)

            left_visited  = left_cell  in self.visited_cells
            right_visited = right_cell in self.visited_cells

            if left_visited and not right_visited:
                action = 'turn_right'
            elif right_visited and not left_visited:
                action = 'turn_left'
            else:
                # Both visited (or neither): alternate from the last wall-turn
                # to ensure we never return the same action twice in a row.
                if self.last_wall_turn == 'turn_left':
                    action = 'turn_right'
                else:
                    action = 'turn_left'

            self.last_wall_turn = action
        else:
            action = 'move_forward'
            self.last_wall_turn = None

        self.last_action = action
        return action


# ---------------------------------------------------------------------------
# Greedy (random) agent — kept for backwards compatibility with simulator.py
# ---------------------------------------------------------------------------
class GreedyGridAgent:
    """A simple agent that wanders randomly to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


# ---------------------------------------------------------------------------
# Search Agent (Practical 03) — BFS / DFS / UCS offline planner
# ---------------------------------------------------------------------------
class SearchAgent:
    """Goal-based agent that plans a complete path using uninformed search.

    The agent supports two calling conventions for the search methods:

    1. **Turn-based** (used by ``sense_and_act`` and the visual simulation):
       ``bfs_search(start_pos, start_facing, goal_pos, walls, grid_size)``
       Actions: ``'move_forward'``, ``'turn_left'``, ``'turn_right'``

    2. **Coordinate-based** (used by ``test_suite.py``):
       ``bfs_search(start_pos, goal_pos, walls, grid_size)``
       Actions: ``'Up'``, ``'Down'``, ``'Left'``, ``'Right'``

    The two conventions are distinguished automatically by checking whether
    the third positional argument is a string (facing direction) or a
    collection (walls list).
    """

    FACING_OFFSETS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
    TURN_LEFT  = {'Up': 'Left',  'Left': 'Down',  'Down': 'Right', 'Right': 'Up'}
    TURN_RIGHT = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left',  'Left': 'Up'}
    MOVE_DIRS  = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.plan         = []
        self.active_algo  = 'AStar'  # 'BFS' | 'DFS' | 'UCS' | 'AStar'
        self.rel_pos      = (0, 0)
        self.facing       = 'Up'
        self.last_action  = None

    # ------------------------------------------------------------------
    # Internal state update (turn-based simulation)
    # ------------------------------------------------------------------
    def _update_state(self):
        if self.last_action == 'move_forward':
            dx, dy = self.FACING_OFFSETS[self.facing]
            self.rel_pos = (self.rel_pos[0] + dx, self.rel_pos[1] + dy)
        elif self.last_action == 'turn_left':
            self.facing = self.TURN_LEFT[self.facing]
        elif self.last_action == 'turn_right':
            self.facing = self.TURN_RIGHT[self.facing]

    # ------------------------------------------------------------------
    # Compass → turn-based action converter (used by A* in the visual sim)
    # ------------------------------------------------------------------
    def _compass_to_turn_actions(self, compass_path: list, start_facing: str) -> list:
        """Convert a list of compass directions into turn-based actions.

        compass_path  : ['Up', 'Right', 'Down', ...]
        start_facing  : current heading of the agent, e.g. 'Up'

        Returns a flat list of 'turn_left', 'turn_right', 'move_forward'.
        """
        actions = []
        current_facing = start_facing

        for compass_dir in compass_path:
            # Rotate until the agent faces the desired compass direction
            turns = 0
            while current_facing != compass_dir and turns < 4:
                # Choose the shorter rotation (at most 1 step needed each time)
                # Determine whether left or right is closer
                left_facing  = self.TURN_LEFT[current_facing]
                right_facing = self.TURN_RIGHT[current_facing]

                if left_facing == compass_dir:
                    actions.append('turn_left')
                    current_facing = left_facing
                else:
                    actions.append('turn_right')
                    current_facing = right_facing
                turns += 1

            actions.append('move_forward')

        return actions

    # ------------------------------------------------------------------
    # Step 1.1 — Heuristic functions (Practical 04)
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal) -> int:
        """h(n) = |x1 - x2| + |y1 - y2|
        Admissible for 4-way grid movement: never over-estimates real cost."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal) -> float:
        """h(n) = sqrt((x1-x2)^2 + (y1-y2)^2)
        Straight-line distance — admissible but less informed than manhattan
        for a 4-way grid (actual path can never be shorter than straight line)."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ------------------------------------------------------------------
    # Perception → action loop
    # ------------------------------------------------------------------
    def sense_and_act(self, percept: dict) -> str:
        self._update_state()

        if percept['food_here']:
            self.last_action = 'suck'
            return 'suck'

        if not self.plan:
            closest_food = min(
                percept['all_food'],
                key=lambda f: abs(self.rel_pos[0] - f[0]) + abs(self.rel_pos[1] - f[1])
            )
            walls_set = set(map(tuple, percept['walls']))
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(self.rel_pos, self.facing, closest_food,
                                            walls_set, percept['grid_size'])
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(self.rel_pos, self.facing, closest_food,
                                            walls_set, percept['grid_size'])
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(self.rel_pos, self.facing, closest_food,
                                            walls_set, percept['grid_size'])
            elif self.active_algo == 'AStar':
                # Step 1.3 — A* integration (Practical 04)
                # A* returns compass directions ('Up'/'Down'/'Left'/'Right').
                # The visual simulation needs turn-based actions, so we convert
                # the compass path into the equivalent turn_left/turn_right/move_forward sequence.
                compass_path = self.astar_search(
                    self.rel_pos, closest_food,
                    walls_set, percept['grid_size'],
                    heuristic_type='manhattan'
                )
                self.plan = self._compass_to_turn_actions(compass_path, self.facing)

        if self.plan:
            action = self.plan.pop(0)
            self.last_action = action
            return action

        # Fallback: rotate until a plan can be made
        self.last_action = 'turn_right'
        return 'turn_right'

    # ------------------------------------------------------------------
    # Helper: detect which calling convention is being used
    # ------------------------------------------------------------------
    @staticmethod
    def _is_coord_call(third_arg):
        """Return True when called as bfs_search(pos, goal, walls, grid_size)."""
        return not isinstance(third_arg, str)

    # ------------------------------------------------------------------
    # Coordinate-based search helpers (used by test_suite.py)
    # Actions: 'Up', 'Down', 'Left', 'Right'
    # ------------------------------------------------------------------
    def _bfs_coord(self, start_pos, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        frontier  = deque([(start_pos, [])])
        reached   = {start_pos}

        while frontier:
            pos, path = frontier.popleft()

            if pos == goal_pos:
                return path

            for direction, (dx, dy) in self.MOVE_DIRS.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                    continue
                if new_pos in walls_set:
                    continue
                if new_pos not in reached:
                    reached.add(new_pos)
                    frontier.append((new_pos, path + [direction]))

        return []   # goal unreachable

    def _dfs_coord(self, start_pos, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        frontier  = [(start_pos, [])]
        reached   = set()

        while frontier:
            pos, path = frontier.pop()
            if pos in reached:
                continue
            reached.add(pos)

            if pos == goal_pos:
                return path

            for direction, (dx, dy) in self.MOVE_DIRS.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                    continue
                if new_pos in walls_set:
                    continue
                if new_pos not in reached:
                    frontier.append((new_pos, path + [direction]))

        return []

    def _ucs_coord(self, start_pos, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        counter   = itertools.count()
        frontier  = []
        heapq.heappush(frontier, (0, next(counter), start_pos, []))
        reached   = set()

        while frontier:
            cost, _, pos, path = heapq.heappop(frontier)
            if pos in reached:
                continue
            reached.add(pos)

            if pos == goal_pos:
                return path

            for direction, (dx, dy) in self.MOVE_DIRS.items():
                new_pos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                    continue
                if new_pos in walls_set:
                    continue
                if new_pos not in reached:
                    heapq.heappush(frontier, (cost + 1, next(counter), new_pos, path + [direction]))

        return []

    # ------------------------------------------------------------------
    # Turn-based search helpers (used by sense_and_act / visual sim)
    # Actions: 'move_forward', 'turn_left', 'turn_right'
    # ------------------------------------------------------------------
    def _bfs_turn(self, start_pos, start_facing, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        frontier  = deque([(start_pos, start_facing, [])])
        reached   = {(start_pos, start_facing)}

        while frontier:
            pos, facing, path = frontier.popleft()

            if pos == goal_pos:
                return path

            for action in ['move_forward', 'turn_left', 'turn_right']:
                new_pos    = pos
                new_facing = facing

                if action == 'move_forward':
                    dx, dy  = self.FACING_OFFSETS[facing]
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                        continue
                    if new_pos in walls_set:
                        continue
                elif action == 'turn_left':
                    new_facing = self.TURN_LEFT[facing]
                elif action == 'turn_right':
                    new_facing = self.TURN_RIGHT[facing]

                state = (new_pos, new_facing)
                if state not in reached:
                    reached.add(state)
                    frontier.append((new_pos, new_facing, path + [action]))

        return []

    def _dfs_turn(self, start_pos, start_facing, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        frontier  = [(start_pos, start_facing, [])]
        reached   = set()

        while frontier:
            pos, facing, path = frontier.pop()

            state = (pos, facing)
            if state in reached:
                continue
            reached.add(state)

            if pos == goal_pos:
                return path

            for action in ['turn_right', 'turn_left', 'move_forward']:
                new_pos    = pos
                new_facing = facing

                if action == 'move_forward':
                    dx, dy  = self.FACING_OFFSETS[facing]
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                        continue
                    if new_pos in walls_set:
                        continue
                elif action == 'turn_left':
                    new_facing = self.TURN_LEFT[facing]
                elif action == 'turn_right':
                    new_facing = self.TURN_RIGHT[facing]

                new_state = (new_pos, new_facing)
                if new_state not in reached:
                    frontier.append((new_pos, new_facing, path + [action]))

        return []

    def _ucs_turn(self, start_pos, start_facing, goal_pos, walls, grid_size):
        walls_set = set(map(tuple, walls))
        counter   = itertools.count()
        frontier  = []
        heapq.heappush(frontier, (0, next(counter), start_pos, start_facing, []))
        reached   = set()

        while frontier:
            cost, _, pos, facing, path = heapq.heappop(frontier)

            state = (pos, facing)
            if state in reached:
                continue
            reached.add(state)

            if pos == goal_pos:
                return path

            for action in ['move_forward', 'turn_left', 'turn_right']:
                new_pos    = pos
                new_facing = facing
                action_cost = 1

                if action == 'move_forward':
                    dx, dy  = self.FACING_OFFSETS[facing]
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if not (0 <= new_pos[0] < grid_size[0] and 0 <= new_pos[1] < grid_size[1]):
                        continue
                    if new_pos in walls_set:
                        continue
                elif action == 'turn_left':
                    new_facing = self.TURN_LEFT[facing]
                elif action == 'turn_right':
                    new_facing = self.TURN_RIGHT[facing]

                new_state = (new_pos, new_facing)
                if new_state not in reached:
                    heapq.heappush(frontier,
                                   (cost + action_cost, next(counter),
                                    new_pos, new_facing, path + [action]))

        return []

    # ------------------------------------------------------------------
    # Step 1.2 — A* Search (Practical 04)
    # Coordinate-based: returns compass actions 'Up'/'Down'/'Left'/'Right'
    # f(n) = g(n) + h(n)
    # ------------------------------------------------------------------
    def astar_search(self, start_pos, goal_pos, walls, grid_size,
                     heuristic_type: str = 'manhattan') -> list:
        """A* search from start_pos to goal_pos.

        Parameters
        ----------
        start_pos      : (x, y) tuple — starting cell
        goal_pos       : (x, y) tuple — target cell
        walls          : iterable of (x, y) wall positions
        grid_size      : (width, height) of the grid
        heuristic_type : 'manhattan' (default) or 'euclidean'

        Returns a list of compass-direction actions, or [] if unreachable.
        """
        # Choose heuristic
        if heuristic_type == 'euclidean':
            h = lambda pos: self.euclidean_distance(pos, goal_pos)
        else:
            h = lambda pos: self.manhattan_distance(pos, goal_pos)

        walls_set = set(map(tuple, walls))
        counter   = itertools.count()   # tie-breaker for equal f-costs

        # Priority queue entries: (f_cost, g_cost, tie_break, current_pos, path_taken)
        start_h = h(start_pos)
        frontier = []
        heapq.heappush(frontier, (start_h, 0, next(counter), start_pos, []))

        reached_states = set()

        while frontier:
            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)

            # Goal test
            if current_pos == goal_pos:
                return path_taken

            # Skip already-explored states
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            # Expand neighbours (4-way movement)
            for direction, (dx, dy) in self.MOVE_DIRS.items():
                neighbor = (current_pos[0] + dx, current_pos[1] + dy)

                # Bounds check
                if not (0 <= neighbor[0] < grid_size[0] and
                        0 <= neighbor[1] < grid_size[1]):
                    continue
                # Wall check
                if neighbor in walls_set:
                    continue
                # Already explored
                if neighbor in reached_states:
                    continue

                g_new = g_cost + 1          # uniform step cost
                h_new = h(neighbor)
                f_new = g_new + h_new

                heapq.heappush(frontier,
                               (f_new, g_new, next(counter),
                                neighbor, path_taken + [direction]))

        return []   # goal is unreachable

    # ------------------------------------------------------------------
    # Public search methods — auto-dispatch on calling convention
    # ------------------------------------------------------------------
    def bfs_search(self, start_pos, arg2, arg3, arg4, arg5=None):
        """BFS search.

        Turn-based:   bfs_search(start_pos, start_facing, goal_pos, walls, grid_size)
        Coord-based:  bfs_search(start_pos, goal_pos,     walls,    grid_size)
        """
        if arg5 is None:
            # Coord-based: (start_pos, goal_pos, walls, grid_size)
            return self._bfs_coord(start_pos, arg2, arg3, arg4)
        else:
            # Turn-based: (start_pos, start_facing, goal_pos, walls, grid_size)
            return self._bfs_turn(start_pos, arg2, arg3, arg4, arg5)

    def dfs_search(self, start_pos, arg2, arg3, arg4, arg5=None):
        """DFS search.

        Turn-based:   dfs_search(start_pos, start_facing, goal_pos, walls, grid_size)
        Coord-based:  dfs_search(start_pos, goal_pos,     walls,    grid_size)
        """
        if arg5 is None:
            return self._dfs_coord(start_pos, arg2, arg3, arg4)
        else:
            return self._dfs_turn(start_pos, arg2, arg3, arg4, arg5)

    def ucs_search(self, start_pos, arg2, arg3, arg4, arg5=None):
        """UCS search.

        Turn-based:   ucs_search(start_pos, start_facing, goal_pos, walls, grid_size)
        Coord-based:  ucs_search(start_pos, goal_pos,     walls,    grid_size)
        """
        if arg5 is None:
            return self._ucs_coord(start_pos, arg2, arg3, arg4)
        else:
            return self._ucs_turn(start_pos, arg2, arg3, arg4, arg5)
        

