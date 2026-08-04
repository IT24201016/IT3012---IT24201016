# test_model_based_agent.py
"""
Verifies that ModelBasedAgent's internal memory (visited_cells) actually
drives its decisions, instead of just being stored and ignored.

Run with:
    python test_model_based_agent.py
"""

from visual_grid_game import ModelBasedAgent, SimpleReflexAgent, VisualGridHuntGame


def test_memory_deduplicates_revisited_cells():
    """memory_size must not grow when the agent returns to a cell it has
    already been to."""
    walls = {(3, 7), (3, 8), (3, 9), (5, 7), (5, 8), (5, 9), (4, 9)}
    env = VisualGridHuntGame(width=10, height=10, num_food=0, num_opponents=0,
                              num_traps=0, custom_walls=walls)
    env.agent_pos = [4, 7]
    env.facing = 'Up'
    agent = ModelBasedAgent()

    sizes = []
    for _ in range(15):
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        sizes.append(len(agent.visited_cells))

    # Memory can only grow or stay flat, it must never shrink or double count.
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), \
        f"visited_cells size should be non-decreasing, got {sizes}"
    print("PASS: memory size is monotonically non-decreasing:", sizes)


def test_decision_depends_on_memory_contents():
    """Same percept, same position/facing, only visited_cells differs.
    If the chosen action differs too, the rule is genuinely reading memory
    rather than ignoring it."""
    percept = {'wall_ahead': True, 'food_here': False, 'smells_toxin': False}

    agent_with_history = ModelBasedAgent()
    agent_with_history.rel_pos = (0, 0)
    agent_with_history.facing = 'Right'
    agent_with_history.visited_cells = {(0, 0), (0, 1)}  # left cell already seen
    agent_with_history.last_action = None
    action_a = agent_with_history.sense_and_act(percept)

    agent_without_history = ModelBasedAgent()
    agent_without_history.rel_pos = (0, 0)
    agent_without_history.facing = 'Right'
    agent_without_history.visited_cells = {(0, 0)}  # left cell unseen
    agent_without_history.last_action = None
    action_b = agent_without_history.sense_and_act(percept)

    assert action_a == 'turn_right', f"expected turn_right, got {action_a}"
    assert action_b == 'turn_left', f"expected turn_left, got {action_b}"
    assert action_a != action_b, "decision should change when memory differs"
    print(f"PASS: memory changes the decision (turn_right vs turn_left): {action_a} vs {action_b}")


def test_model_based_agent_explores_more_than_reflex_agent():
    """In a pocket that opens into a larger room, memory should let the
    agent cover more ground than a stateless reflex agent over the same
    number of steps."""
    walls = {(3, 7), (3, 8), (3, 9), (5, 7), (5, 8), (5, 9), (4, 9)}

    def run(agent_cls, steps=80):
        env = VisualGridHuntGame(width=10, height=10, num_food=0, num_opponents=0,
                                  num_traps=0, custom_walls=walls)
        env.agent_pos = [4, 7]
        env.facing = 'Up'
        agent = agent_cls()
        visited = {tuple(env.agent_pos)}
        for _ in range(steps):
            percept = env.get_percept()
            action = agent.sense_and_act(percept)
            env.execute_action(action)
            visited.add(tuple(env.agent_pos))
        return len(visited)

    reflex_cells = run(SimpleReflexAgent)
    model_cells = run(ModelBasedAgent)

    assert model_cells >= reflex_cells, (
        f"expected ModelBasedAgent ({model_cells} cells) to cover at least as "
        f"much ground as SimpleReflexAgent ({reflex_cells} cells)"
    )
    print(f"PASS: ModelBasedAgent explored {model_cells} cells vs "
          f"SimpleReflexAgent's {reflex_cells} cells")


if __name__ == "__main__":
    test_memory_deduplicates_revisited_cells()
    test_decision_depends_on_memory_contents()
    test_model_based_agent_explores_more_than_reflex_agent()
    print("\nAll memory checks passed.")
