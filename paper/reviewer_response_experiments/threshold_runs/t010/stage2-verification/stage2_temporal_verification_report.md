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
| Any-Regular2960 | 6 | 0 | 0 | 0 | 0 | 0 | 5 | 1 |
| BeefSupreme2 | 6 | 0 | 0 | 0 | 0 | 1 | 5 | 1 |
| BtcKing1111 | 55 | 2 | 1 | 0 | 5 | 1 | 26 | 26 |
| Covetoast | 6 | 1 | 0 | 0 | 1 | 2 | 4 | 1 |
| Different_Walrus_574 | 21 | 0 | 0 | 0 | 0 | 0 | 8 | 13 |
| DuckDuckMosss | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| Local_Doubt_4029 | 28 | 0 | 0 | 0 | 0 | 0 | 19 | 9 |
| Odd-Following-247 | 55 | 0 | 0 | 0 | 0 | 3 | 40 | 15 |
| Sweet-Hat-7946 | 10 | 0 | 1 | 0 | 2 | 0 | 2 | 7 |
| Wsemenske | 21 | 0 | 0 | 0 | 3 | 0 | 12 | 9 |
| abercrombezie | 66 | 1 | 0 | 0 | 2 | 1 | 14 | 51 |
| harvested | 28 | 1 | 0 | 1 | 0 | 0 | 4 | 23 |
| iPurchaseBitcoin | 21 | 1 | 0 | 1 | 0 | 0 | 20 | 0 |
| lol_camis | 253 | 4 | 0 | 0 | 10 | 3 | 68 | 181 |
| onebtcisonebtc | 36 | 0 | 0 | 0 | 2 | 0 | 13 | 23 |
| sirspeedy99 | 66 | 0 | 0 | 0 | 5 | 1 | 21 | 45 |
| skydiver19 | 91 | 0 | 1 | 0 | 2 | 1 | 33 | 57 |
| tkwh | 120 | 2 | 2 | 0 | 9 | 3 | 86 | 30 |
| tzacPACO | 15 | 0 | 3 | 1 | 2 | 1 | 12 | 0 |
| vnielz | 21 | 1 | 0 | 0 | 1 | 1 | 10 | 10 |

## Key Pairs

- BtcKing1111: BtcKing1111 <-> DavidGunn454 | moderate_temporal_sync / moderate_review | same_post=3 | <5min=0 | <30min=1 | median_delay=65.3min | co_neg=0.000
- BtcKing1111: DavidGunn454 <-> KaleidoscopeShot8153 | strong_temporal_sync / fragile_long_median | same_post=8 | <5min=1 | <30min=1 | median_delay=180.2min | co_neg=0.217
- BtcKing1111: DavidGunn454 <-> anonymoushusky11 | strong_temporal_sync / moderate_review | same_post=2 | <5min=1 | <30min=1 | median_delay=38.7min | co_neg=0.000
- Covetoast: Covetoast <-> Potential_Initial903 | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=1.8min | co_neg=0.447
- Sweet-Hat-7946: Caesars7Hills <-> Sweet-Hat-7946 | moderate_temporal_sync / moderate_review | same_post=3 | <5min=0 | <30min=1 | median_delay=49.3min | co_neg=0.218
- abercrombezie: Significant_Book1672 <-> abercrombezie | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=0.1min | co_neg=0.202
- harvested: NectarineDirect936 <-> harvested | strong_temporal_sync / robust | same_post=17 | <5min=2 | <30min=4 | median_delay=83.2min | co_neg=0.335
- iPurchaseBitcoin: Safe-Painter-9618 <-> iPurchaseBitcoin | strong_temporal_sync / robust | same_post=4 | <5min=1 | <30min=2 | median_delay=48.3min | co_neg=0.000
- lol_camis: AmphibianAway8217 <-> _Zzik_ | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=2.8min | co_neg=0.000
- lol_camis: AmphibianAway8217 <-> lol_camis | strong_temporal_sync / moderate_review | same_post=2 | <5min=1 | <30min=1 | median_delay=97.1min | co_neg=0.231
- lol_camis: Dismal-Grapefruit966 <-> choppedyota | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=4.4min | co_neg=0.000
- lol_camis: _Zzik_ <-> lol_camis | strong_temporal_sync / moderate_review | same_post=3 | <5min=1 | <30min=1 | median_delay=59.7min | co_neg=0.338
- skydiver19: CatsGotANosebleed <-> skydiver19 | moderate_temporal_sync / fragile_long_median | same_post=3 | <5min=0 | <30min=1 | median_delay=230.4min | co_neg=0.200
- tkwh: mookizee <-> tkwh | moderate_temporal_sync / moderate_review | same_post=4 | <5min=0 | <30min=2 | median_delay=101.1min | co_neg=0.261
- tkwh: 753UDKM <-> buffwhoppulus | moderate_temporal_sync / moderate_review | same_post=3 | <5min=0 | <30min=1 | median_delay=53.8min | co_neg=0.000
- tkwh: AdventurousSwim1381 <-> mikeso623 | strong_temporal_sync / moderate_review | same_post=2 | <5min=1 | <30min=2 | median_delay=6.5min | co_neg=0.000
- tkwh: buffwhoppulus <-> mikeso623 | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=1.2min | co_neg=0.000
- tzacPACO: Rent_South <-> tzacPACO | moderate_temporal_sync / robust | same_post=3 | <5min=0 | <30min=2 | median_delay=29.0min | co_neg=0.216
- tzacPACO: Romando1 <-> tzacPACO | moderate_temporal_sync / moderate_review | same_post=2 | <5min=0 | <30min=2 | median_delay=54.5min | co_neg=0.220
- tzacPACO: Romando1 <-> btcgib | moderate_temporal_sync / fragile_long_median | same_post=3 | <5min=0 | <30min=1 | median_delay=160.9min | co_neg=0.218
- vnielz: BitcoinFreedom1776 <-> turdturd1 | strong_temporal_sync / fragile_single_event | same_post=1 | <5min=1 | <30min=1 | median_delay=1.1min | co_neg=0.000
