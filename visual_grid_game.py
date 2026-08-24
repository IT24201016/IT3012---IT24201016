# visual_grid_game.py
import random
import tkinter as tk


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    # Directional helpers: the agent has a facing direction, and "ahead" is
    # always relative to that facing (this is what makes wall_ahead/food_here
    # meaningful sensor readings instead of leaked global coordinates).
    FACING_OFFSETS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
    TURN_LEFT = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}
    TURN_RIGHT = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left', 'Left': 'Up'}

    def __init__(
        self,
        width=10,
        height=10,
        num_food=10,
        num_opponents=2,
        num_traps=5,
        custom_walls=None,
    ):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.facing = 'Up'  # Agent's own heading (proprioception, not a global-position leak)

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        # Toxic traps are part of the external environment. Keep them away from
        # the starting cell, walls, food, and opponents so every object has an
        # unambiguous initial location.
        blocked_positions = (
            {(0, 0)}
            | self.walls
            | self.food_positions
            | {tuple(opponent) for opponent in self.opponents}
        )
        available_positions = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in blocked_positions
        ]
        if num_traps > len(available_positions):
            raise ValueError(
                f"Cannot place {num_traps} traps: only "
                f"{len(available_positions)} safe cells are available."
            )
        self.toxic_traps = set(random.sample(available_positions, num_traps))

        self.score = 0
        self.steps = 0
        self.collision = False

    def get_percept(self) -> dict:
        """Partially observable sensor reading: local booleans relative to the
        agent's current facing direction only. No global coordinates are
        exposed (agent_pos / opponent_positions are gone) so the agent cannot
        cheat its way around blind spots."""
        dx, dy = self.FACING_OFFSETS[self.facing]
        ahead = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        ahead_in_bounds = 0 <= ahead[0] < self.width and 0 <= ahead[1] < self.height
        wall_ahead = (not ahead_in_bounds) or (ahead in self.walls)

        return {
            'wall_ahead': wall_ahead,
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'smells_toxin': tuple(self.agent_pos) in self.toxic_traps,
            'facing': self.facing,
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions),
            'grid_size': (self.width, self.height),
            'walls': list(self.walls),
            'all_food': list(self.food_positions),
        }

    def execute_action(self, action: str):
        """Actions are relative to the agent's facing: turn_left, turn_right,
        move_forward, suck. This mirrors a real robot, which only ever knows
        'turn' and 'go forward', never absolute compass moves."""
        self.steps += 1

        if action == 'turn_left':
            self.facing = self.TURN_LEFT[self.facing]
        elif action == 'turn_right':
            self.facing = self.TURN_RIGHT[self.facing]
        elif action == 'suck':
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20
        elif action == 'move_forward':
            dx, dy = self.FACING_OFFSETS[self.facing]
            new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
            in_bounds = 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height

            if not in_bounds or tuple(new_pos) in self.walls:
                self.score -= 5  # bumped into a wall/edge, stayed put
            else:
                self.agent_pos = new_pos

        tuple_pos = tuple(self.agent_pos)
        if tuple_pos in self.toxic_traps:
            self.score -= 15

        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


class SimpleReflexAgent:
    """Step 1.2: pure condition-action rules, no memory of past percepts.
    This is expected to get trapped in corners/U-shaped walls because it has
    no way to remember it already tried turning left here before."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'suck'
        if percept['wall_ahead']:
            return 'turn_left'
        return 'move_forward'


class ModelBasedAgent:
    """Step 1.3: keeps an internal model (transition model) of its own
    position and heading, built purely from the actions it has chosen to
    take (it still never sees the environment's real agent_pos). This lets
    it recognise "I've already been here" and break out of loops that trap
    the SimpleReflexAgent."""

    FACING_OFFSETS = VisualGridHuntGame.FACING_OFFSETS
    TURN_LEFT = VisualGridHuntGame.TURN_LEFT
    TURN_RIGHT = VisualGridHuntGame.TURN_RIGHT

    def __init__(self):
        self.rel_pos = (0, 0)          # believed position, relative to start
        self.facing = 'Up'             # believed heading, relative to start
        self.visited_cells = {(0, 0)}
        self.last_action = None
        self.last_percept = None

    def _update_state(self, percept: dict):
        """Transition model: given the action we just took, predict how our
        internal position/heading changed. We only ever call move_forward
        when the previous percept said wall_ahead is False, so it's safe to
        assume the move succeeded. Sensor model: also record the raw percept
        that triggered this update, for reference/debugging."""
        self.last_percept = percept

        if self.last_action == 'move_forward':
            dx, dy = self.FACING_OFFSETS[self.facing]
            self.rel_pos = (self.rel_pos[0] + dx, self.rel_pos[1] + dy)
        elif self.last_action == 'turn_left':
            self.facing = self.TURN_LEFT[self.facing]
        elif self.last_action == 'turn_right':
            self.facing = self.TURN_RIGHT[self.facing]

        self.visited_cells.add(self.rel_pos)

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)

        if percept['food_here']:
            action = 'suck'
        elif percept['wall_ahead']:
            # Check memory before deciding which way to turn: if we already
            # visited the cell to our left, turning left just repeats the
            # loop the SimpleReflexAgent gets stuck in, so try right instead.
            left_facing = self.TURN_LEFT[self.facing]
            ldx, ldy = self.FACING_OFFSETS[left_facing]
            left_cell = (self.rel_pos[0] + ldx, self.rel_pos[1] + ldy)

            action = 'turn_right' if left_cell in self.visited_cells else 'turn_left'
        else:
            action = 'move_forward'

        self.last_action = action
        return action


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, num_traps=5, walls=None,
                 agent_class=SimpleReflexAgent):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(
            width=width,
            height=height,
            num_food=num_food,
            num_opponents=num_opponents,
            num_traps=num_traps,
            custom_walls=walls,
        )
        self.agent = agent_class()

        # Leave room below the canvas for the label, button, taskbar, and
        # window title bar so they never get pushed off the bottom of the
        # screen. winfo_screenheight() reports the FULL monitor height, it
        # does not subtract the taskbar, so we reserve extra margin for that.
        screen_h = root.winfo_screenheight()
        reserved_for_controls = 220
        max_canvas_dim = min(600, screen_h - reserved_for_controls)
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()
        self._center_window(root)

    @staticmethod
    def _center_window(root):
        root.update_idletasks()
        width = root.winfo_reqwidth()
        height = root.winfo_reqheight()

        # Assume a ~60px taskbar and leave a safety margin so the button
        # can never end up hidden behind it.
        taskbar_estimate = 60
        usable_height = root.winfo_screenheight() - taskbar_estimate

        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = max(20, (usable_height // 2) - (height // 2))
        if y + height > usable_height:
            y = max(20, usable_height - height)

        root.geometry(f"{width}x{height}+{x}+{y}")
        root.resizable(False, False)

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                # Only draw text if cell is large enough
                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        # Draw each toxic trap as a purple diamond.
        for tx, ty in self.env.toxic_traps:
            center_x = (tx + 0.5) * self.cell_size
            center_y = (self.env.height - ty - 0.5) * self.cell_size
            radius = self.cell_size * 0.32
            self.canvas.create_polygon(
                center_x,
                center_y - radius,
                center_x + radius,
                center_y,
                center_x,
                center_y + radius,
                center_x - radius,
                center_y,
                fill="#9333ea",
                outline="#581c87",
                width=2,
            )

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    from agent import SearchAgent
    root = tk.Tk()
    # Practical 04: Use SearchAgent with A* (active_algo='AStar') for optimised pathfinding.
    agent = SearchAgent()
    agent.active_algo = 'AStar'
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0,
                       agent_class=lambda: agent)
    root.mainloop()
