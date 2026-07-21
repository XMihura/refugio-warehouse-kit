import unittest

import warehouse_api
from warehouse.actions import Action
from warehouse.layout import CellType
from warehouse.observation import Observation


class WarehouseApiTests(unittest.TestCase):
    def test_public_exports_are_stable(self) -> None:
        self.assertEqual(warehouse_api.Action, Action)
        self.assertEqual(warehouse_api.CellType, CellType)
        self.assertEqual(warehouse_api.Observation, Observation)

    def test_all_contains_only_public_agent_contract_symbols(self) -> None:
        self.assertEqual(
            set(warehouse_api.__all__),
            {
                "ActFunction",
                "Action",
                "CellType",
                "GridView",
                "Observation",
                "Position",
            },
        )

    def test_typical_agent_import_pattern(self) -> None:
        from warehouse_api import Action as PublicAction
        from warehouse_api import Observation as PublicObservation

        def act(_observation: PublicObservation) -> PublicAction:
            return PublicAction.WAIT

        self.assertEqual(act.__annotations__["return"], PublicAction)


if __name__ == "__main__":
    unittest.main()
