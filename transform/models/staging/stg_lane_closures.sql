-- One row per Caltrans lane closure (jobs/lane_closures.py writes one row per closure per day it
-- was listed; this combines them).
--
--   started_at   when the crew reported the closure in place (code 1097); null = never confirmed
--   active_to    when it was over: the crew's end report (1098); else, if it dropped out of the feed,
--                the last time it was listed; else (still listed) the latest feed time
--   route_number freeway number ("I-405" -> 405); direction_letter N/S/E/W (B = both directions)

with days as (
    select * from {{ source('bronze', 'caltrans_lane_closures') }}
),

merged as (
    select
        closure_key,
        max_by(district, last_seen)                                                       as district,
        max_by(route, last_seen)                                                          as route,
        max_by(direction, last_seen)                                                      as direction,
        max_by(begin_lat, last_seen)                                                      as begin_lat,
        max_by(begin_lon, last_seen)                                                      as begin_lon,
        max_by(coalesce(end_lat, begin_lat), last_seen)                                   as end_lat,
        max_by(coalesce(end_lon, begin_lon), last_seen)                                   as end_lon,
        max_by(facility, last_seen)                                                       as facility,
        max_by(closure_type, last_seen)                                                   as closure_type,
        max_by(work_type, last_seen)                                                      as work_type,
        max_by(try_cast(regexp_extract(lanes_closed, '[0-9]+') as integer), last_seen)    as lanes_closed,
        max_by(try_cast(regexp_extract(total_lanes, '[0-9]+') as integer), last_seen)     as total_lanes,
        max_by(planned_start, last_seen)                                                  as planned_start,
        max_by(planned_end, last_seen)                                                    as planned_end,
        max(started_at)                                                                   as started_at,
        max(ended_at)                                                                     as ended_at,
        max(cancelled_at)                                                                 as cancelled_at,
        min(first_seen)                                                                   as first_seen,
        max(last_seen)                                                                    as last_seen
    from days
    group by closure_key
),

feed as (
    select max(last_seen) as feed_last_seen from days
)

select
    m.*,
    try_cast(regexp_extract(m.route, '([0-9]+)', 1) as integer)                            as route_number,
    upper(substr(m.direction, 1, 1))                                                      as direction_letter,
    case
        when m.ended_at is not null then m.ended_at
        when m.last_seen < f.feed_last_seen - interval '15' minute then m.last_seen
        else f.feed_last_seen
    end                                                                                   as active_to
from merged m
cross join feed f
