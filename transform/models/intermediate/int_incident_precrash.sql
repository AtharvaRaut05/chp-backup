-- Traffic conditions before each incident: one row per incident, all measured BEFORE
-- the crash (window ends log_lag_minutes before CHP's log time), so they're safe to use
-- as model inputs.
--
--   site_*      the sensor at/just upstream of the incident (queue rank 1)
--   queue_*     the queue zone (incident to upstream_miles upstream)
--   approach_*  upstream_miles to approach_miles upstream: demand arriving at the queue
--   merge_* / exit_*  ramp and connector volume entering / leaving the queue zone
--   *_trend     speed in the last 15 min minus the 15 min before that (negative = slowing)
--   is_measurable  at least one healthy queue sensor reported before the crash; incidents
--                  without one can't have their backup measured and are left out of labels

{% set lag = var('log_lag_minutes') %}
{% set win = var('precrash_minutes') %}

with pre as (
    select *
    from {{ ref('int_incident_traffic') }}
    where minutes_from_log >= -({{ lag }} + {{ win }}) and minutes_from_log < -{{ lag }}
      and is_observed
),

recent_split as (
    select
        *,
        minutes_from_log >= -({{ lag }} + 15) as is_recent
    from pre
)

select
    log_id,

    -- at the incident site
    avg(avg_speed)     filter (where role = 'queue' and role_rank = 1)                      as site_speed,
    avg(avg_occupancy) filter (where role = 'queue' and role_rank = 1)                      as site_occupancy,
    avg(total_flow)    filter (where role = 'queue' and role_rank = 1)                      as site_flow,
    max(pre_ratio)     filter (where role = 'queue' and role_rank = 1)                      as site_ratio_to_typical,
    avg(avg_speed) filter (where role = 'queue' and role_rank = 1 and is_recent)
      - avg(avg_speed) filter (where role = 'queue' and role_rank = 1 and not is_recent)    as site_speed_trend,

    -- queue zone
    avg(avg_speed)     filter (where role = 'queue')                                        as queue_speed,
    min(avg_speed)     filter (where role = 'queue')                                        as queue_min_speed,
    avg(avg_occupancy) filter (where role = 'queue')                                        as queue_occupancy,
    avg(avg_speed) filter (where role = 'queue' and is_recent)
      - avg(avg_speed) filter (where role = 'queue' and not is_recent)                      as queue_speed_trend,
    count(distinct station_id) filter (where role = 'queue')                                as queue_sensors,
    count(distinct station_id) filter (where role = 'queue') > 0                            as is_measurable,

    -- approach zone (traffic about to arrive)
    avg(avg_speed)     filter (where role = 'approach')                                     as approach_speed,
    avg(total_flow)    filter (where role = 'approach')                                     as approach_flow_per_sensor,
    avg(total_flow) filter (where role = 'approach' and is_recent)
      - avg(total_flow) filter (where role = 'approach' and not is_recent)                  as approach_flow_trend,

    -- traffic merging in / exiting within the queue zone (vehicles per 5 min, all ramps summed)
    sum(total_flow) filter (where role = 'merge') / {{ win // 5 }}                          as merge_flow,
    sum(total_flow) filter (where role = 'merge' and station_type = 'FF') / {{ win // 5 }}  as connector_merge_flow,
    sum(total_flow) filter (where role = 'exit') / {{ win // 5 }}                           as exit_flow,
    count(distinct station_id) filter (where role = 'merge')                                as merge_points

from recent_split
group by log_id
