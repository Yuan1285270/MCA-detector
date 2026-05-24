# Stage 2 Temporal Synchrony Verification

This report verifies Stage 1 candidate groups by checking whether group members appear in the same Reddit thread within short time windows.

Labels:

- `strong_temporal_sync`: at least one co-comment event within 5 minutes
- `moderate_temporal_sync`: at least two co-comment events within 30 minutes
- `weak_temporal_overlap`: same thread overlap, but no short-window synchrony
- `no_temporal_sync`: no same-thread overlap in the local comments file

Confidence:

- `robust`: repeated or multi-post short-window synchrony
- `moderate_review`: useful timing evidence that still needs manual review
- `fragile_single_event`: label rests on one short-window event
- `fragile_long_median`: has short-window evidence, but typical delay is long

Formal evidence:

- `verification_label`: temporal synchrony strength
- `temporal_confidence`: reliability calibration for temporal synchrony

Deprecated:

- `text_fingerprint_distance`: retained as a compatibility column, but intentionally left blank. TF-IDF text distance is treated as topic-noise in this single-topic dataset and is not used as verification evidence.
- `account_lifecycle_overlap`: retained as a compatibility column, but intentionally left blank. Lifecycle overlap is treated as topic/activity-window noise and is not used as verification evidence.

## Group Summary

| group_seed | pairs | strong | moderate | robust | moderate review | fragile | weak | no sync |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Any-Regular2960 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Covetoast | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| Local_Doubt_4029 | 21 | 0 | 0 | 0 | 0 | 0 | 13 | 8 |
| Wsemenske | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| abercrombezie | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| harvested | 3 | 1 | 0 | 1 | 0 | 0 | 2 | 0 |
| iPurchaseBitcoin | 15 | 1 | 0 | 1 | 0 | 0 | 14 | 0 |
| lol_camis | 3 | 1 | 0 | 0 | 2 | 0 | 1 | 1 |
| onebtcisonebtc | 21 | 0 | 0 | 0 | 0 | 0 | 7 | 14 |
| tkwh | 15 | 0 | 0 | 0 | 0 | 0 | 11 | 4 |
| tzacPACO | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

## Key Pairs

- Covetoast: Covetoast <-> Potential_Initial903 | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=1.8min | co_neg=0.447
- harvested: NectarineDirect936 <-> harvested | strong_temporal_sync / robust | same_post=17 | <5min=2 | <30min=4 | median_delay=83.2min | co_neg=0.335
- iPurchaseBitcoin: Safe-Painter-9618 <-> iPurchaseBitcoin | strong_temporal_sync / robust | same_post=4 | <5min=1 | <30min=2 | median_delay=48.3min | co_neg=0.000
- lol_camis: _Zzik_ <-> lol_camis | strong_temporal_sync / moderate_review | same_post=3 | <5min=1 | <30min=1 | median_delay=59.7min | co_neg=0.338
