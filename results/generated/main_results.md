# Reported results

Event-macro averages over the 12 BerlinI evaluation events.

## Standard-objective comparison

| System | Parameters (M) | H RMSE | NSE | Wet RMSE | Wet NSE | Peak error | CSI .01 | CSI .10 | CSI .30 | CSI .50 | Target ACE | Target WIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CANO (standard objective) | 0.277 | 0.0195 | 0.9733 | 0.0236 | 0.9716 | 0.3724 | 0.7640 | 0.8414 | 0.9029 | 0.7678 | 0.0714 | 0.0121 |
| DNO-3 (standard objective) | 10.815 | 0.0288 | 0.9424 | 0.0352 | 0.9369 | 0.2614 | 0.7137 | 0.7771 | 0.8373 | 0.6482 | 0.0280 | 0.0219 |
| FNO3D (standard objective) | 10.627 | 0.0294 | 0.9396 | 0.0304 | 0.9521 | 0.1864 | 0.6679 | 0.7802 | 0.8955 | 0.7063 | 0.0157 | 0.0166 |
| U-Net3D (standard objective) | 5.650 | 0.0380 | 0.8703 | 0.0474 | 0.8613 | 0.6357 | 0.7270 | 0.7553 | 0.7712 | 0.6529 | 0.0645 | 0.0355 |

## CANO training-objective ablation

| System | Parameters (M) | H RMSE | NSE | Wet RMSE | Wet NSE | Peak error | CSI .01 | CSI .10 | CSI .30 | CSI .50 | Target ACE | Target WIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CANO (standard objective) | 0.277 | 0.0195 | 0.9733 | 0.0236 | 0.9716 | 0.3724 | 0.7640 | 0.8414 | 0.9029 | 0.7678 | 0.0714 | 0.0121 |
| CANO (selection-matched control) | 0.277 | 0.0195 | 0.9732 | 0.0237 | 0.9714 | 0.3728 | 0.7638 | 0.8412 | 0.9027 | 0.7677 | 0.0715 | 0.0121 |
| CANO (peak-aware objective) | 0.277 | 0.0164 | 0.9798 | 0.0197 | 0.9795 | 0.0380 | 0.7567 | 0.8541 | 0.9295 | 0.8261 | 0.0428 | 0.0074 |

NSE and wet-domain NSE are core point-prediction metrics. Target ACE and Target WIS summarize target-aligned uncertainty calibration.
