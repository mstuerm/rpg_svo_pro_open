# Camera-0 post-tracking frontend health, matching PAMPC SAC training.
# Header stamp is the processed camera image time, not publication time.
Header header
# False when no fresh frontend frame is available. Do not feed invalid data
# to a policy; normalized values below are finite placeholders in that case.
bool telemetry_available
uint32 tracked_feature_count
# Survival uses nonnegative track_id_vec_ IDs, NOT map-point IDs.
# NaN if there is no previous tracked ID set (including after reset).
float64 tracked_landmark_survival_ratio
# Median finite geometric pixel residual over tracked features with landmarks.
# NaN when no residual exists.
float64 median_reprojection_error_px
float64 median_track_age_frames
# Exact learner order: count/100, survival, residual/2, age/10.
# Clips: [0,1.5], [0,1], [0,1.5], [0,1.5].
# Undefined survival -> 0; undefined residual -> 1.5.
float64[4] normalized
