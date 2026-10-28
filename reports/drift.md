### Data drift: `2026-06` vs reference `2026-01`

**Drift detected** in 3 feature(s): txn_count_24h, customer_age, avg_amount_30d (threshold 0.2, max PSI 0.308)

Fraud rate: 2.017% (reference) -> 2.890% (current)

| feature | PSI | status |
|---|---:|---|
| txn_count_24h | 0.3081 | 🔴 drift |
| customer_age | 0.2076 | 🔴 drift |
| avg_amount_30d | 0.2031 | 🔴 drift |
| card_present | 0.1442 | 🟡 warning |
| account_age_days | 0.1204 | 🟡 warning |
| amount | 0.1094 | 🟡 warning |
| merchant_category | 0.1074 | 🟡 warning |
| hour | 0.0651 | 🟢 ok |
| device_change | 0.0310 | 🟢 ok |
| is_foreign | 0.0280 | 🟢 ok |
| distance_from_home_km | 0.0169 | 🟢 ok |
