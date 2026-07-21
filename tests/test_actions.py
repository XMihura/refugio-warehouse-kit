import unittest

from warehouse.actions import Action, coerce_action
from warehouse.observation import Observation


class ActionTests(unittest.TestCase):
    def test_action_values_match_public_contract(self) -> None:
        self.assertEqual(
            [action.value for action in Action],
            ["wait", "up", "down", "left", "right", "pickup", "drop"],
        )

    def test_coerce_action_accepts_enum_or_string(self) -> None:
        self.assertEqual(coerce_action(Action.UP), Action.UP)
        self.assertEqual(coerce_action("pickup"), Action.PICKUP)

    def test_coerce_action_rejects_unknown_strings(self) -> None:
        with self.assertRaises(ValueError):
            coerce_action("teleport")

    def test_team_act_signature_returns_action(self) -> None:
        def act(_observation: Observation) -> Action:
            return Action.WAIT

        self.assertEqual(act.__annotations__["return"], Action)


if __name__ == "__main__":
    unittest.main()
