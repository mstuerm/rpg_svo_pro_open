Header      header
float32     processing_time
int32[]     keyframes
int32       num_matches
int32       tracking_quality
int32       stage

# Per-frame map-point reprojection trace. All arrays below are parallel.
# outcome: 0=matched, 1=match failed, 2=not visible, 3=failed-reprojection
# limit, 4=too few observations, 5=grid cell occupied, 6=feature budget,
# 7=no close-view observation, 8=removed by pose-optimization reprojection gate.
# match_result: -1=not attempted; otherwise Matcher::MatchResult ordinal:
# 0=success, 1=score, 2=triangulation, 3=visibility, 4=warp,
# 5=alignment, 6=range, 7=angle, 8=close-view, 9=lock, 10=too-far.
int32[]     landmark_trace_frame_ids
int32[]     landmark_trace_point_ids
int32[]     landmark_trace_reference_frame_ids
int32[]     landmark_trace_reference_feature_indices
uint8[]     landmark_trace_outcomes
int16[]     landmark_trace_match_results
int32[]     landmark_trace_succeeded_reprojections
int32[]     landmark_trace_failed_reprojections
float32[]   landmark_trace_projected_x
float32[]   landmark_trace_projected_y
float32[]   landmark_trace_matched_x
float32[]   landmark_trace_matched_y
float32[]   landmark_trace_match_displacement_px
float32[]   landmark_trace_reprojection_error_px
float32[]   landmark_trace_point_world_x
float32[]   landmark_trace_point_world_y
float32[]   landmark_trace_point_world_z
