# PPT Graph Explanation Assets

Generated from current local pipeline outputs.

## Recommended placement

1. `slide_04_updated_architecture.svg`  
   Put on Slide 4, replacing the old system flow.  
   Purpose: shows the current architecture: LLM -> feature/multi-graph -> MCA seed -> Stage 1 -> Stage 2 -> outputs -> demo site.

2. `slide_19_multigraph_edge_density.svg`  
   Put in the "Why not one graph?" section, around Slide 19.  
   Purpose: real edge counts explain why count/trigger are dense and noisy while co-negative is sparse/high-signal.

3. `slide_20_count_vs_conegative_harvested.svg`  
   Put right after the edge-density slide.  
   Purpose: harvested example shows co-target/context neighbors are broad, while co-negative selects a smaller direct expansion set.

4. `slide_21_conegative_threshold_harvested.svg`  
   Put on the co-negative explanation slide.  
   Purpose: explains `co_negative >= 0.20` using actual harvested Tier 1 pairs and shared negative targets.

5. `slide_36_mca_seed_expansion_pipeline.svg`  
   Put on the "Why not MCA alone?" slide.  
   Purpose: MCA only ranks seeds; expansion and temporal verification turn account suspicion into group investigation.

6. `slide_47_temporal_confidence_harvested_vs_jg.svg`  
   Put in Stage 2 / temporal confidence section.  
   Purpose: harvested and JG87919 both have co-negative structure, but confidence separates robust timing from fragile timing.

7. `slide_47_temporal_confidence_harvested_vs_odd.svg`  
   Recommended replacement for the JG87919 version in the paper/PPT.  
   Purpose: compares two official top-20 seed groups. harvested has Stage 1 structure plus robust temporal evidence, while Odd-Following-247 has strong Stage 1 structure but no reliable Stage 2 timing evidence.

## Key speaking point

Co-negative works because it compresses many negative comment edges into a shared-target pattern. It is stricter than count/co-target because it asks whether accounts repeatedly oppose the same targets, not merely whether they appear in the same discussion space.

## Architecture recommendation

Yes, the overall architecture slide should be changed. The old PPT flow treats behavior matrix, candidate ranking, role/group analysis as a flat sequence. The updated flow should show two separate layers:

- analysis pipeline: raw data -> LLM -> features/graphs -> MCA seed ranking -> Stage 1 expansion -> Stage 2 verification -> evidence tables
- presentation layer: demo site reads generated evidence tables only
