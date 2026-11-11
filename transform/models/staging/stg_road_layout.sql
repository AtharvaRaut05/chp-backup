-- Road layout at each incident from OpenStreetMap (jobs/road_layout.py). One row per incident;
-- if a rerun wrote it twice, keep the newest. match_m is null when no freeway was found nearby.

select
    log_id,
    max_by(match_m, dt)             as match_m,
    max_by(road_heading_deg, dt)    as road_heading_deg,
    max_by(road_lanes, dt)          as road_lanes,
    max_by(lane_drop_ahead, dt)     as lane_drop_ahead,
    max_by(exits_ahead_1mi, dt)     as exits_ahead_1mi,
    max_by(merges_behind_1mi, dt)   as merges_behind_1mi,
    max_by(dist_next_exit_mi, dt)   as dist_next_exit_mi,
    max_by(dist_prev_merge_mi, dt)  as dist_prev_merge_mi,
    max_by(curve_deg, dt)           as curve_deg,
    max_by(maxspeed_mph, dt)        as maxspeed_mph,
    max_by(on_bridge, dt)           as on_bridge
from {{ source('bronze', 'road_layout') }}
group by log_id
