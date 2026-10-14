-- Traffic conditions before each incident: one row per incident, all measured BEFORE
-- the crash (window ends log_lag_minutes before CHP's log time), so they're safe to use
-- as model inputs.
--
--   site_*      the closest healthy sensor at/just upstream of the incident
--   queue_*     the queue zone (incident to upstream_miles upstream)
--   approach_*  upstream_miles to approach_miles upstream: demand arriving at the queue
--   merge_* / exit_*  ramp and connector volume entering / leaving the queue zone
--   *_trend     speed in the last 15 min minus the 15 min before that (negative = slowing)
--   *_lanes     mainline lanes at the site sensor / averaged over the approach sensors (for
--               capacity: how many lanes a crash leaves open, how much traffic per lane)
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

site as (
    -- closest queue sensor that was actually reporting before the crash
    select log_id, min_by(station_id, role_rank) as site_station_id
    from pre
    where role = 'queue'
    group by log_id
),

recent_split as (
    select
        p.*,
        st.lanes,
        p.station_id = s.site_station_id       as is_site,
        p.minutes_from_log >= -({{ lag }} + 15) as is_recent
    from pre p
    left join site s on s.log_id = p.log_id
    left join {{ ref('int_incident_stations') }} st on st.log_id = p.log_id and st.station_id = p.station_id
)

select
    log_id,
    max(station_id) filter (where is_site)                         as site_station_id,

    -- at the incident site
    avg(avg_speed)     filter (where is_site)                      as site_speed,
    avg(avg_occupancy) filter (where is_site)                      as site_occupancy,
    avg(total_flow)    filter (where is_site)                      as site_flow,
    max(pre_ratio)     filter (where is_site)                      as site_ratio_to_typical,
    max(lanes)         filter (where is_site)                      as site_lanes,
    avg(avg_speed) filter (where is_site and is_recent)
      - avg(avg_speed) filter (where is_site and not is_recent)    as site_speed_trend,

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
    avg(lanes)         filter (where role = 'approach')                                     as approach_lanes,
    avg(total_flow) filter (where role = 'approach' and is_recent)
      - avg(total_flow) filter (where role = 'approach' and not is_recent)                  as approach_flow_trend,

    -- traffic merging in / exiting within the queue zone (vehicles per 5 min, all ramps summed),
    -- divided by the number of 5-minute slots that actually had readings
    sum(total_flow) filter (where role = 'merge')
      / nullif(count(distinct minutes_from_log) filter (where role = 'merge'), 0)              as merge_flow,
    sum(total_flow) filter (where role = 'merge' and station_type = 'FF')
      / nullif(count(distinct minutes_from_log) filter (where role = 'merge' and station_type = 'FF'), 0) as connector_merge_flow,
    sum(total_flow) filter (where role = 'exit')
      / nullif(count(distinct minutes_from_log) filter (where role = 'exit'), 0)               as exit_flow,
    count(distinct station_id) filter (where role = 'merge')                                as merge_points

from recent_split
group by log_id
