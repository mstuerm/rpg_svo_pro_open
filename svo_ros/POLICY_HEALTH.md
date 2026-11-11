# Policy health telemetry

The SVO ROS node publishes `svo_msgs/PolicyHealth` on its private
`~policy_health` topic (for a node named `/kingfisher/svo`, this resolves to
`/kingfisher/svo/policy_health`). This is a separate, small message: subscribing
does not require subscribing to dense images or enabling landmark tracing.
Both simulation and the drone use this same publisher. Rebuild `svo_msgs` and
`svo_ros`, then source the rebuilt workspace on publishers and consumers.

```bash
rostopic type /kingfisher/svo/policy_health
rostopic echo /kingfisher/svo/policy_health
rostopic hz /kingfisher/svo/policy_health
```

`header.stamp` is the camera-0 image timestamp. Health is evaluated after frontend
processing, before the optional info/dense subscriber gates, once per fresh
frame. It is not an IMU-rate or policy-rate signal. A consumer should hold the
latest sample between camera frames and check timestamp freshness together with
`telemetry_available` and the existing SVO validity signal. No messages during
paused image processing imply no fresh telemetry; do not reuse stale health.
This message alone does not authorize controller handover.

## Exact training contract

Matches `RpgSvoShadow._perception_health` and
`PAMPCStateEnv._set_svo_health` with
`sac_health_use_landmark_survival_ratio: true`, default scales 100, 2 px,
10 frames. Uses only camera 0, including with stereo input.

| Raw field | Definition | `normalized` index |
| --- | --- | --- |
| `tracked_feature_count` | Tracked landmarks excluding fixed/map points, plus matched corner/edgelet seeds, counted per feature slot | 0: count / 100, clipped [0, 1.5] |
| `tracked_landmark_survival_ratio` | Intersection of current and previous nonnegative tracked **track IDs**, divided by previous ID count | 1: clipped [0, 1] |
| `median_reprojection_error_px` | Median finite Euclidean pixel residual after projecting each tracked slot's 3-D landmark with the frontend pose and camera model; excludes seeds without landmarks | 2: error / 2, clipped [0, 1.5] |
| `median_track_age_frames` | Median consecutive-frame age of unique nonnegative tracked IDs; new IDs start at 1 and disappearing IDs are discarded | 3: age / 10, clipped [0, 1.5] |

The track population and projection intentionally match `svo_pybind.latest_features`.
Track IDs are `track_id_vec_`, not map-point IDs or reprojection-attempt counters.
Matched seeds with nonnegative track IDs participate in the age/survival set,
just as in the Python training reducer. Repeated IDs count once for age/survival,
but tracked feature count and residual median use feature slots.

Raw survival is NaN when no preceding ID set exists, and raw reprojection median
is NaN if no finite residual exists. Normalized substitutes are respectively 0
and 1.5. Empty age is 0. The `normalized` array always contains four finite values
and is ready to append after the 16-value base MPCC observation. An invalid
sample contains placeholders `[0, 0, 1.5, 0]` and must not be used for inference.

History is maintained even without subscribers. Reset, fresh start, stationary
calibration, frontend pause/reset, missing frames, and backwards image timestamps
clear it. Duplicate publication of the same fresh image does not increment ages
or publish a second sample. Explicit reset invalidates history immediately; the
next processed paused frame publishes unavailable telemetry. Consumers must also
honor the existing reset/validity gate while no new image is processed.

## Validation

The reducer has a standalone test (also registered with CTest when catkin testing
is enabled):

```bash
g++ -std=c++11 -Wall -Wextra -Werror -I svo_ros/include \
  svo_ros/test/test_policy_health.cpp -o /tmp/test_policy_health
/tmp/test_policy_health
python3 svo_ros/test/test_policy_health_parity.py /path/to/flightpy/src/PAMPC
```

The parity test requires NumPy. It executes the actual training Python methods
on 200 deterministic generated frames and compares all normalized values against
the C++ reducer, including resets, disappearing tracks, duplicated/negative IDs,
missing residuals, clipping and empty frames. It does not validate camera
calibration or real flight timing. Check those in simulation and shadow logs.
