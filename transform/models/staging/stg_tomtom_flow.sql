-- Live speed lookups made when each new freeway incident first appeared (src/tomtom/speed_lookup.py).
-- One row per incident: the first lookup (an incident is only looked up once, but a retried
-- invocation can write it twice).

select
    log_id,
    min(requested_at)                                    as requested_at,
    min_by(frc, requested_at)                            as frc,
    min_by(current_speed_kmh, requested_at)              as current_speed_kmh,
    min_by(free_flow_speed_kmh, requested_at)            as free_flow_speed_kmh,
    min_by(confidence, requested_at)                     as confidence,
    min_by(road_closure, requested_at)                   as road_closure,
    min_by(segment_bearing_deg, requested_at)            as segment_bearing_deg
from {{ source('bronze', 'tomtom_flow') }}
group by log_id
