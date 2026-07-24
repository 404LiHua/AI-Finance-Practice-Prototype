import unittest

import pandas as pd

from stage_d.rolling_origin import build_fold_assignments, generate_fold_boundaries


class RollingOriginTest(unittest.TestCase):
    def panel(self) -> pd.DataFrame:
        dates = pd.date_range("2022-06-03", periods=50, freq="W-FRI")
        rows = []
        for stock in ("000001.SZ", "000002.SZ"):
            for index, date in enumerate(dates):
                target_date = dates[index + 1] if index + 1 < len(dates) else pd.NaT
                rows.append({
                    "stock_code": stock,
                    "trade_date": date,
                    "target_date": target_date,
                    "target_return": 0.01 if pd.notna(target_date) else None,
                    "history_weeks_available": index + 1,
                    "cross_section_eligible": True,
                })
        return pd.DataFrame(rows)

    def test_three_folds_advance_and_purge(self) -> None:
        panel = self.panel()
        folds = generate_fold_boundaries(
            pd.Index(panel["trade_date"].unique()), 3, 24, 6, 6, 1
        )
        self.assertEqual([fold.fold_id for fold in folds], ["D_RO_01", "D_RO_02", "D_RO_03"])
        for fold in folds:
            self.assertLess(pd.Timestamp(fold.train_end), pd.Timestamp(fold.purge_start))
            self.assertLess(pd.Timestamp(fold.purge_end), pd.Timestamp(fold.validation_start))

    def test_assignments_have_no_train_validation_overlap(self) -> None:
        panel = self.panel()
        folds = generate_fold_boundaries(
            pd.Index(panel["trade_date"].unique()), 3, 24, 6, 6, 1
        )
        assignments, metadata = build_fold_assignments(panel, folds, 12, 2)
        self.assertEqual(len(metadata), 3)
        for fold_id, frame in assignments.groupby("fold_id"):
            train = frame[frame["split"] == "train"]
            validation = frame[frame["split"] == "validation"]
            self.assertLess(train["target_date"].max(), validation["trade_date"].min())
            self.assertEqual(validation["stock_code"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
