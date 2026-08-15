from __future__ import annotations

"""Synthetic contract tests for the future-T2 preconsumption gate."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = Path(__file__).resolve().with_name("audit_wp23_future_t2_input_contract_v1.py")
PROTOCOL = ROOT / "governance" / "WP23_FUTURE_T2_INPUT_PRECONSUMPTION_CONTRACT_FREEZE_20260815.json"
RG3 = ["momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d", "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d", "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d", "zero_volume_fraction_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d"]
RG2 = ["capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag", "log_total_shares_change_at_event", "log_tradable_shares_change_at_event", "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled", "capital_history_missing_flag", "market_tradable_fraction", "market_eligible_fraction", "market_small_cap_fraction", "industry_tradable_fraction", "industry_eligible_fraction", "log1p_industry_member_count", "graph_mean_absolute_change", "graph_intra_industry_weight_fraction", "graph_mean_nonself_out_degree_scaled", "graph_max_nonself_out_degree_scaled"]


class WP23ContractTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> dict[str, Path]:
        dates = pd.date_range("2026-07-03", periods=12, freq="7D")
        stocks = [f"{index:06d}.SZ" for index in range(1, 301)]
        keys = [(date, stock) for date in dates for stock in stocks]
        origin = pd.DataFrame({"trade_date": dates, "cutoff_at_utc": dates + pd.Timedelta(hours=7), "cutoff_rule_id": "SYNTHETIC_FRIDAY_CLOSE_V1"})
        universe = pd.DataFrame(keys, columns=["trade_date", "stock_code"]); universe["eligible"] = True; universe["membership_effective_at"] = universe["trade_date"] - pd.Timedelta(days=1)
        rg3 = pd.DataFrame(keys, columns=["trade_date", "stock_code"]); rg3["source_trade_date"] = rg3["trade_date"]
        for position, name in enumerate(RG3): rg3[name] = float(position + 1) / 100.0
        rg2 = pd.DataFrame(keys, columns=["trade_date", "stock_code"])
        for position, name in enumerate(RG2): rg2[name] = float(position + 1) / 100.0
        for name in ("capital_effective_date_asof", "membership_state_date", "graph_state_date"): rg2[name] = rg2["trade_date"]
        scale = pd.DataFrame(keys, columns=["trade_date", "stock_code"]); scale["market_volatility_4w"] = 0.2
        paths = {}
        for name, frame in (("origins", origin), ("universe", universe), ("rg3", rg3), ("rg2", rg2), ("scale", scale)):
            path = root / f"{name}.csv"; frame.to_csv(path, index=False); paths[name] = path
        paths["protocol"] = PROTOCOL
        return paths

    def run_auditor(self, paths: dict[str, Path], output: Path) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(AUDITOR), *sum(([f"--{name.replace('_', '-')}", str(path)] for name, path in paths.items()), []), "--output-root", str(output)]
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_valid_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wp23_valid_") as directory:
            root = Path(directory); paths = self.make_bundle(root); completed = self.run_auditor(paths, root / "audit_pass")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            decision = json.loads((root / "audit_pass" / "WP23_FUTURE_T2_INPUT_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["status"], "PASS_READY_FOR_PREDICTION_SEAL_AND_NEW_ONE_TIME_LABEL_AUTHORIZATION")
            self.assertEqual(decision["origin_count"], 12)
            self.assertEqual(decision["eligible_stock_count"], 300)

    def test_forbidden_label_column_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wp23_forbidden_") as directory:
            root = Path(directory); paths = self.make_bundle(root); scale = pd.read_csv(paths["scale"]); scale["target_label"] = 0; scale.to_csv(paths["scale"], index=False)
            completed = self.run_auditor(paths, root / "audit_fail")
            self.assertNotEqual(completed.returncode, 0)
            decision = json.loads((root / "audit_fail" / "WP23_FUTURE_T2_INPUT_PRECONSUMPTION_AUDIT.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["status"], "FAIL_CLOSED_FUTURE_T2_INPUT_PRECONSUMPTION")
            self.assertTrue(any("forbidden_columns" in failure for failure in decision["failures"]))


if __name__ == "__main__":
    unittest.main()


