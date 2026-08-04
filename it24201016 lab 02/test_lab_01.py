import random
import unittest

from visual_grid_game import VisualGridHuntGame


class ToxicTrapTests(unittest.TestCase):
    def setUp(self):
        random.seed(3012)
        self.game = VisualGridHuntGame(
            width=6,
            height=6,
            num_food=4,
            num_opponents=2,
            num_traps=5,
        )

    def test_traps_are_placed_safely(self):
        self.assertEqual(len(self.game.toxic_traps), 5)
        self.assertNotIn((0, 0), self.game.toxic_traps)
        self.assertTrue(self.game.toxic_traps.isdisjoint(self.game.walls))
        self.assertTrue(self.game.toxic_traps.isdisjoint(self.game.food_positions))
        self.assertTrue(
            self.game.toxic_traps.isdisjoint(map(tuple, self.game.opponents))
        )

    def test_percept_reports_toxin_on_current_cell(self):
        trap = next(iter(self.game.toxic_traps))
        self.game.agent_pos = list(trap)
        self.assertTrue(self.game.get_percept()["smells_toxin"])

        self.game.agent_pos = [0, 0]
        self.assertFalse(self.game.get_percept()["smells_toxin"])

    def test_stepping_on_trap_deducts_fifteen_points(self):
        self.game.agent_pos = [0, 0]
        self.game.toxic_traps = {(1, 0)}
        self.game.food_positions.discard((1, 0))
        self.game.score = 0

        self.game.execute_action("Right")

        self.assertEqual(self.game.agent_pos, [1, 0])
        self.assertEqual(self.game.score, -15)

    def test_impossible_trap_count_is_rejected(self):
        with self.assertRaises(ValueError):
            VisualGridHuntGame(
                width=2,
                height=2,
                num_food=0,
                num_opponents=0,
                num_traps=4,
                custom_walls=set(),
            )


if __name__ == "__main__":
    unittest.main()
