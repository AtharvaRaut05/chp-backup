{{ config(materialized='view') }}
-- 5-minute sensor readings. A view, so queries only read what they need.
-- is_observed = false means PeMS filled the reading in from neighbors instead of measuring it.
-- Ramps and connectors (lane_type OR/FR/FF) usually report flow but no speed.

select
    station_id,
    lane_type,
    district,
    dt,
    ts,
    total_flow,
    avg_occupancy,
    avg_speed,
    pct_observed,
    pct_observed >= {{ var('min_pct_observed') }} as is_observed
from {{ source('bronze', 'pems_station_5min') }}
