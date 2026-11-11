-- Lane closures in place where and when each incident was logged (Caltrans Lane Closure System).
--
--   closure_nearby          a closure the crew had reported in place (and not yet over) on the same
--                           freeway and direction, within about 1.5 km of the crash
--   closure_lanes_closed    most lanes closed by such a closure
--   closure_full            ... one of them closed the whole road
--   planned_closure_nearby  an approved closure scheduled for that time and place whose start was
--                           never confirmed (some crews don't report it)
--
-- Known when the incident is logged. Null (not 0) for incidents outside the time the feed was being
-- saved: the feed only shows current closures, so there is no history before collection started.
-- "Near" = inside an ellipse around the closure's begin and end points: the distances from the crash
-- to both ends add up to at most the closure's length + 3 km.

with incidents as (
    select log_id, log_time, freeway, direction, lat, lon
    from {{ ref('stg_chp_incidents') }}
    where is_traffic and freeway is not null and direction is not null and lat is not null
),

closures as (
    select *, great_circle_distance(begin_lat, begin_lon, end_lat, end_lon) as length_km
    from {{ ref('stg_lane_closures') }}
    where begin_lat is not null and route_number is not null
),

coverage as (
    select min(first_seen) as start_at, max(last_seen) as end_at from {{ ref('stg_lane_closures') }}
),

matches as (
    select
        i.log_id,
        c.lanes_closed,
        lower(c.closure_type) like '%full%'                                                as is_full,
        c.started_at is not null and i.log_time >= c.started_at and i.log_time < c.active_to as in_place,
        c.started_at is null and c.cancelled_at is null
            and i.log_time >= c.planned_start and i.log_time < c.planned_end                as planned
    from incidents i
    join closures c
        on c.route_number = i.freeway
       and (c.direction_letter = i.direction or c.direction_letter = 'B')
       and great_circle_distance(i.lat, i.lon, c.begin_lat, c.begin_lon)
         + great_circle_distance(i.lat, i.lon, c.end_lat, c.end_lon) <= c.length_km + 3
)

select
    i.log_id,
    if(covered, coalesce(bool_or(m.in_place), false), null)                                   as closure_nearby,
    if(covered, coalesce(max(m.lanes_closed) filter (where m.in_place), 0), null)              as closure_lanes_closed,
    if(covered, coalesce(bool_or(m.in_place and m.is_full), false), null)                      as closure_full,
    if(covered, coalesce(bool_or(m.planned), false), null)                                     as planned_closure_nearby
from (
    select i.*, i.log_time >= cv.start_at and i.log_time <= cv.end_at as covered
    from incidents i
    cross join coverage cv
) i
left join matches m on m.log_id = i.log_id
group by i.log_id, i.covered
