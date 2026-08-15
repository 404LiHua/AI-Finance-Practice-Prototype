import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_panel_v2.py"
SPEC = importlib.util.spec_from_file_location("panel_v2", MODULE_PATH)
PANEL_V2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PANEL_V2
assert SPEC.loader is not None
SPEC.loader.exec_module(PANEL_V2)


class PanelV2Tests(unittest.TestCase):
    def base(self):
        return pd.DataFrame({
            "trade_date": pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"]),
            "stock_code": ["000001.SZ"] * 3,
            "model_close": [10.0, 10.5, 10.2],
            "model_volume_hands": [100.0, 0.0, 100.0],
            "csmar_special_status": ["normal"] * 3,
        })

    def static_inputs(self):
        basic = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "listing_date": pd.to_datetime(["2020-01-05"]),
            "listing_status_snapshot": ["D"],
        })
        company = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "company_listing_date": pd.to_datetime(["2020-01-05"]),
        })
        return basic, company

    def test_listing_and_delisting_are_point_in_time(self):
        basic, company = self.static_inputs()
        events = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "listing_status_event_effective_date": pd.to_datetime(["2020-01-17"]),
            "listing_status_announced_at": pd.to_datetime(["2020-01-15"]),
            "listing_status_observable_at": pd.to_datetime(["2020-01-17"]),
            "listing_status_before": ["正常上市"],
            "listing_status_after": ["终止上市"],
            "listing_status_change_type": ["AX"],
            "listing_status_source_sha256": ["x"],
            "listing_status_source_row": [2],
        })
        result = PANEL_V2.add_lifecycle(self.base(), basic, company, events)
        self.assertFalse(result.iloc[0].universe_member_pit)
        self.assertTrue(result.iloc[1].universe_member_pit)
        self.assertFalse(result.iloc[2].universe_member_pit)
        self.assertTrue(result.iloc[2].is_delisted_asof)

    def test_suspended_stock_remains_member_but_is_not_eligible(self):
        basic, company = self.static_inputs()
        result = PANEL_V2.add_lifecycle(self.base(), basic, company, pd.DataFrame())
        result = PANEL_V2.add_trading_state(result)
        suspended = result.iloc[1]
        self.assertTrue(suspended.universe_member_pit)
        self.assertTrue(suspended.is_suspended)
        self.assertFalse(suspended.model_eligible_pit)
        self.assertEqual(suspended.trade_state, "suspended")

    def test_dense_panel_retains_missing_bar(self):
        base = self.base().copy()
        base["split"] = "train"
        second = base.copy()
        second["stock_code"] = "000002.SZ"
        second = second.drop(index=1)
        dense = PANEL_V2.dense_panel(pd.concat([base, second], ignore_index=True))
        self.assertEqual(len(dense), 6)
        dense = dense[dense.stock_code.eq("000002.SZ")].reset_index(drop=True)
        basic, company = self.static_inputs()
        basic = pd.concat([basic, basic.assign(stock_code="000002.SZ")], ignore_index=True)
        company = pd.concat([company, company.assign(stock_code="000002.SZ")], ignore_index=True)
        dense = PANEL_V2.add_lifecycle(dense, basic, company, pd.DataFrame())
        dense = PANEL_V2.add_trading_state(dense)
        missing = dense.iloc[1]
        self.assertTrue(missing.is_no_weekly_bar)
        self.assertEqual(missing.trade_state, "no_weekly_bar")
        self.assertFalse(missing.model_eligible_pit)
        self.assertEqual(missing.split, "train")

    def test_capital_and_adjustment_never_use_future_event(self):
        capital = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "capital_effective_date": pd.to_datetime(["2020-01-17"]),
            "capital_change_type": ["01100"],
            "total_shares": [200.0],
            "tradable_a_shares": [150.0],
            "capital_source_sha256": ["c"],
            "capital_source_row": [2],
        })
        adjust = pd.DataFrame({
            "stock_code": ["000001.SZ"],
            "adjust_factor_effective_date": pd.to_datetime(["2020-01-17"]),
            "forward_adjust_factor": [0.8],
            "back_adjust_factor": [1.2],
            "adjust_factor": [1.2],
            "adjust_factor_source_file": ["a.csv"],
            "adjust_factor_source_sha256": ["a"],
        })
        result, ledger = PANEL_V2.add_capital_and_adjustment(self.base(), capital, adjust)
        self.assertTrue(pd.isna(result.iloc[1].total_shares_asof))
        self.assertEqual(result.iloc[2].total_shares_asof, 200.0)
        self.assertEqual(result.iloc[2].adjust_factor_asof, 1.2)
        self.assertTrue(result.iloc[2].corporate_action_this_week)
        self.assertEqual(len(ledger), 2)


if __name__ == "__main__":
    unittest.main()
