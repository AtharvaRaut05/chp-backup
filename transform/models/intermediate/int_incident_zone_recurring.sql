-- Normal (recurring) congestion along the whole backup zone, from OTHER days' history.
-- int_incident_recurring describes the one sensor at the crash; a stretch a mile back that is
-- normally jammed at this hour matters too, because that's where the queue has to grow.
--
--   zone_typical_min_now / _in_30 / _in_60  slowest queue-zone sensor (0-5 mi upstream): its typical
--                                           speed now, 30 and 60 min later / its free-flow speed
--   zone_typical_mean_now                   the same, averaged over the queue-zone sensors
--   zone_share_congested_now                share of queue-zone sensors normally below 70% of free flow now
--   approach_typical_mean_now               average over approach-zone sensors (5-10 mi upstream)
--
-- Typical speed per slot leaves out the incident's own day, as everywhere else. Free-flow speed
-- (90th percentile of a sensor's typical speeds across the day) is computed once per sensor over all
-- days: one day barely moves a 90th percentile, and recomputing it per incident would multiply the cost.
-- All of it is known before the crash, so it's safe as a model input.

{% set min_days = var('min_profile_days') %}

with incidents as (
    select
        i.log_id,
        cast(i.incident_day as varchar)  as incident_dt,
        {{ day_type('i.log_local') }}    as day_type,
        {{ slot_of_day('i.log_local') }} as log_slot
    from (
        select log_id, incident_day, log_time at time zone 'America/Los_Angeles' as log_local
        from {{ ref('stg_chp_incidents') }}
    ) i
    join {{ ref('int_incident_precrash') }} p on p.log_id = i.log_id
    where p.is_measurable
),

sensors as (
    select s.log_id, s.station_id, s.role
    from {{ ref('int_incident_stations') }} s
    join incidents k on k.log_id = s.log_id
    where s.station_type = 'ML' and s.role in ('queue', 'approach')
),

free_flow as (
    select station_id, day_type, approx_percentile(slot_speed, 0.9) as free_flow_speed
    from (
        select station_id, day_type, slot, approx_percentile(speed, 0.5) as slot_speed
        from {{ ref('int_station_profile') }}
        group by station_id, day_type, slot
        having count(*) >= {{ min_days }}
    ) per_slot
    group by station_id, day_type
),

typical as (
    select
        s.log_id,
        s.station_id,
        s.role,
        k.day_type,
        (p.slot - k.log_slot) * 5       as minutes_ahead,
        approx_percentile(p.speed, 0.5) as typical_speed
    from sensors s
    join incidents k on k.log_id = s.log_id
    join {{ ref('int_station_profile') }} p
        on p.station_id = s.station_id
       and p.day_type = k.day_type
       and p.dt <> k.incident_dt
       and p.slot in (k.log_slot, k.log_slot + 6, k.log_slot + 12)
    group by s.log_id, s.station_id, s.role, k.day_type, p.slot - k.log_slot
    having count(*) >= {{ min_days }}
),

ratios as (
    select t.log_id, t.role, t.minutes_ahead, t.typical_speed / f.free_flow_speed as ratio
    from typical t
    join free_flow f on f.station_id = t.station_id and f.day_type = t.day_type
    where f.free_flow_speed > 0
)

select
    log_id,
    min(ratio) filter (where role = 'queue' and minutes_ahead = 0)                  as zone_typical_min_now,
    min(ratio) filter (where role = 'queue' and minutes_ahead = 30)                 as zone_typical_min_in_30,
    min(ratio) filter (where role = 'queue' and minutes_ahead = 60)                 as zone_typical_min_in_60,
    avg(ratio) filter (where role = 'queue' and minutes_ahead = 0)                  as zone_typical_mean_now,
    avg(if(ratio < 0.7, 1.0, 0.0)) filter (where role = 'queue' and minutes_ahead = 0) as zone_share_congested_now,
    avg(ratio) filter (where role = 'approach' and minutes_ahead = 0)               as approach_typical_mean_now
from ratios
group by log_id
