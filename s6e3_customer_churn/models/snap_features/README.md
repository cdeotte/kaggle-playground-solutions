# Snap features: decimal strings and nearest original values of the two charges

No prediction pairs: `build.py` writes three feature tables to `cache/` that many models in `../xgboost`, `../catboost`,
`../lightgbm` and `../neural_nets` read. Run it before those models.

```
python build.py
```
CPU only. Reads `data/` (train.csv, test.csv, WA_Fn-UseC_-Telco-Customer-Churn.csv); nothing has to
run first. `MonthlyCharges` and `TotalCharges` are read as the strings written in the CSV files, so the decimal features
see the values exactly as the generator wrote them. The competition labels are not used.

## `cache/combo.csv` (848,849 rows, 33 columns)

The train rows (train.csv order) followed by the test rows (test.csv order); the row position equals `id`.

| Column | Content |
|---|---|
| `id`, the 19 predictors, `Churn` | as in train.csv / test.csv (`Churn` is empty for the test rows) |
| `tc` | the decimals of TotalCharges as written (the text after the `.`) |
| `tc_len` | number of decimals (1 or 2) |
| `tc0`, `tc1` | first and second decimal digit (`tc1` = -1 when there is one decimal) |
| `mc`, `mc0`, `mc1`, `mc_len` | the same for MonthlyCharges |
| `MonthlyCharges_snap` | MonthlyCharges when the original data has the same string, else the numerically closest original MonthlyCharges (the first one in original row order on ties) |
| `MonthlyCharges_snap_diff` | `MonthlyCharges_snap - MonthlyCharges` (float32) |
| `TotalCharges_snap`, `TotalCharges_snap_diff` | the same for TotalCharges |

Eleven values with three decimals are cut to two before anything is computed: MonthlyCharges of id 151415
(`115.332` -> `115.33`) and TotalCharges of ids 19542, 305615, 319208, 450690, 466283, 492101, 581362 (train) and
653927, 727684, 796527 (test) (the last character is dropped). These fixed values are also the ones written to the
`MonthlyCharges` / `TotalCharges` columns.

## `cache/combo_b.csv` (848,849 rows, 35 columns)

`combo.csv` plus two original-data label rates:

| Column | Content |
|---|---|
| `orig_MonthlyCharges` | churn rate (Yes = 1) of the original rows whose MonthlyCharges equals the row's `MonthlyCharges_snap` |
| `orig_TotalCharges` | churn rate of the original rows whose TotalCharges equals the row's `TotalCharges_snap` |

## `cache/orig.csv` (7,032 rows, 29 columns)

The original rows with a non-blank TotalCharges (the 11 customers with tenure 0 are dropped): `customerID` replaced by
`id` = 848,849 + row number, the 19 predictors, `Churn` (Yes / No), and the decimal strings `mc`, `mc0`, `mc1`,
`mc_len`, `tc`, `tc0`, `tc1`, `tc_len` defined as above (`mc_len` / `tc_len` are empty for values without decimals).
The snap values of `combo.csv` are taken from these rows.

Requirements: pandas, numpy.
