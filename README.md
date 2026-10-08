# MapleFreight Delivery Delay Prediction & Slack Alerts
AML 3303 · Assessment 1

Predict which in-transit shipments will arrive late and alert dispatch in Slack while there is still time to act.

## Problem
MapleFreight moves ~8,000 shipments/week. On-time performance fell from 94% to 88% and delays are only discovered when the customer calls, after service credits and re-delivery costs are already locked in. This project builds a classifier on dispatch-time information and posts a Slack alert for high-risk shipments.

## Dataset
`data/maplefreight_delivery_delay_dataset.csv`: 6,035 raw rows, 23 columns, target `delivered_late` (1 = missed the promised window).

Cleaning
* 32 exact duplicate rows + 3 repeated `shipment_id`s removed -> **6,000 shipments**, 20.6% late
* `service_level` casing normalised (`standard` -> `Standard`)
* 40 weights recorded in grams (171,000-3,000,000) converted to kg
* Missing `weather_condition` -> `Unknown`; missing numerics median-imputed inside the CV pipeline (no leakage)
* `actual_transit_hours` **dropped**: it is recorded on delivery and `actual - scheduled` alone has AUC 0.999 (target leakage). `shipment_id` also dropped.

## Approach
Stratified 80/20 split -> 5-fold stratified CV on the training set to compare models -> threshold chosen on out-of-fold predictions -> single evaluation on the held-out test set. Imbalance is handled with class weights (`balanced` / `scale_pos_weight`); 8 engineered features (pickup-delay ratio, required speed, severe-weather flag, winter flag, etc.).

| Model (CV, training set) | PR-AUC |
|---|---|
| Dummy (always on time) | 0.206 |
| Logistic Regression (baseline, unweighted) | 0.557 |
| **Logistic Regression (balanced) - final** | 0.555 |
| XGBoost (tuned) | 0.544 |
| XGBoost (untuned) | 0.521 |
| Random Forest (balanced) | 0.508 |

The final model is chosen among the imbalance-aware candidates. Tree models did not beat the linear model, so the simpler, more explainable one was kept.

### Test-set results
| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Dummy (always on time) | - | 0.794 | 0.000 | 0.000 | 0.000 |
| Logistic Regression baseline | 0.50 | 0.832 | 0.680 | 0.352 | 0.464 |
| **Logistic Regression balanced (final)** | **0.59** | 0.794 | 0.500 | **0.664** | **0.570** |

Confusion matrix of the final model on the 1,200 test shipments: 164 late caught, 83 late missed, 164 false alarms, 789 correctly left alone. ROC-AUC 0.832, PR-AUC 0.603. Note the final model's accuracy equals the useless dummy's: accuracy is the wrong metric here.

## Chosen alert threshold: 0.59
Picked as the F1-maximising cut-off on **out-of-fold training predictions** (never the test set). A missed late delivery costs service credits and re-delivery; a false alarm costs a dispatcher a few seconds, but too many cause alert fatigue, so F1 balances the two. It flags ~27% of shipments (about 2,200/week at 8,000 shipments), more than can be personally checked, so dispatch should work the list from the highest score down (the top 20% by risk contains ~57% of all late deliveries); raise the threshold for fewer, more certain alerts. It should be re-tuned once the real cost of a late delivery vs. an alert is known.

## Key findings
* **Weather:** storm 46% late, snow 34%, clear 14%.
* **Distance:** 12% late under 300 km, 52% over 1,500 km. More stops and heavier traffic also raise risk steadily.
* **Carrier:** contract owner-operators 36% late vs 13% for MapleFreight's own fleet.
* **Service level:** Economy 27% late vs Express 15%. Customer tier has no effect: Gold accounts are as late as Bronze (~21%).
* **Operations:** a pickup >30 min late -> 27% late at delivery (vs 15% when early); carriers with 3+ lates in the last 30 days run ~28% late.
* **Season:** January-February ~24-25% late vs ~18% in late spring/early autumn.
* Full recommendations are in the last section of the notebook.

## Repository contents
```
MapleFreight_Delivery_Delay_Prediction_c0968946.ipynb   executed notebook
README.md   requirements.txt   .gitignore   .env.example
maplefreight_delivery_delay_dataset.csv
```
No credentials are committed: `.env` is in `.gitignore`, and only `.env.example` (placeholder) is tracked.
