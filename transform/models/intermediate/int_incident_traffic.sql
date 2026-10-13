-- 5-minute traffic at every sensor around each paired incident, from series_before_minutes
-- before to series_after_minutes after CHP's log time, with the expected (no-incident) speed.
--
-- Baseline ("pre-crash, adjusted"):
--   pre_ratio      = actual speed / typical speed, averaged over the pre-crash window
--                    (precrash_minutes long, ending log_lag_minutes before the log time,
--                    because CHP logs a crash a few minutes after it happens)
--   expected_speed = typical speed at that time of day x pre_ratio
-- This starts from the day's real conditions and follows the sensor's normal daily pattern,
-- so a rush-hour slowdown that would have happened anyway isn't blamed on the incident.
-- "Typical" is the median of the same sensor, slot and weekday/weekend on OTHER days (the
-- incident's own day is left out so its slowdown doesn't count as normal). With fewer than
-- min_profile_days such days, expected_speed falls back to the plain pre-crash average.
--
-- Only sensors that were healthy (actually measuring) on that day are included; see
-- int_station_day_health.

with incidents as (
    select log_id, log_time, incident_day
    from {{ ref('stg_chp_incidents') }}
    where pairing_status = 'paired'
),

sensors as (
    select log_id, station_id, station_type, role, role_rank, miles_upstream
    from {{ ref('int_incident_stations') }}
),

readings as (
    select
        s.log_id,
        s.station_id,
        s.station_type,
        s.role,
        s.role_rank,
        s.miles_upstream,
        r.dt,
        r.ts,
        date_diff('minute', i.log_time, r.ts) as minutes_from_log,
        r.avg_speed,
        r.total_flow,
        r.avg_occupancy,
        r.is_observed,
        r.ts at time zone 'America/Los_Angeles' as ts_local
    from sensors s
    join incidents i on i.log_id = s.log_id
    join {{ ref('stg_pems_5min') }} r
        on r.station_id = s.station_id
       and r.dt between cast(date_add('day', -1, i.incident_day) as varchar)
                    and cast(date_add('day', 1, i.incident_day) as varchar)  -- partition pruning
       and r.ts >= i.log_time - interval '{{ var("series_before_minutes") }}' minute
       and r.ts <= i.log_time + interval '{{ var("series_after_minutes") }}' minute
    join {{ ref('int_station_day_health') }} h
        on h.station_id = r.station_id
       and h.dt = r.dt
       and h.is_healthy
),

keyed as (
    select
        *,
        {{ day_type('ts_local') }}    as day_type,
        {{ slot_of_day('ts_local') }} as slot
    from readings
),

-- typical traffic for each (sensor, day, slot) from all OTHER days of the same day type
typical as (
    select
        k.station_id,
        k.dt,
        k.slot,
        approx_percentile(p.speed, 0.5) as typical_speed,
        approx_percentile(p.flow, 0.5)  as typical_flow,
        count(*)                        as profile_days
    from (select distinct station_id, dt, day_type, slot from keyed) k
    join {{ ref('int_station_profile') }} p
        on p.station_id = k.station_id
       and p.day_type = k.day_type
       and p.slot = k.slot
       and p.dt <> k.dt
    group by k.station_id, k.dt, k.slot
),

with_typical as (
    select
        r.*,
        if(t.profile_days >= {{ var('min_profile_days') }}, t.typical_speed) as typical_speed,
        if(t.profile_days >= {{ var('min_profile_days') }}, t.typical_flow)  as typical_flow,
        coalesce(t.profile_days, 0)                                         as profile_days
    from keyed r
    left join typical t
        on t.station_id = r.station_id
       and t.dt = r.dt
       and t.slot = r.slot
),

precrash as (
    select
        log_id,
        station_id,
        avg(avg_speed)     filter (where in_window) as pre_speed,
        avg(typical_speed) filter (where in_window and typical_speed is not null) as pre_typical_speed,
        avg(avg_speed)     filter (where in_window and typical_speed is not null) as pre_speed_matched
    from (
        select
            *,
            is_observed and avg_speed is not null
              and minutes_from_log >= -({{ var('log_lag_minutes') }} + {{ var('precrash_minutes') }})
              and minutes_from_log < -{{ var('log_lag_minutes') }} as in_window
        from with_typical
    )
    group by log_id, station_id
)

select
    t.log_id,
    t.station_id,
    t.station_type,
    t.role,
    t.role_rank,
    t.miles_upstream,
    t.dt,
    t.ts,
    t.minutes_from_log,
    t.avg_speed,
    t.total_flow,
    t.avg_occupancy,
    t.is_observed,
    t.typical_speed,
    t.typical_flow,
    t.profile_days,
    p.pre_speed,
    least(greatest(p.pre_speed_matched / nullif(p.pre_typical_speed, 0), 0.3), 1.3) as pre_ratio,
    case
        when t.typical_speed is not null and p.pre_typical_speed > 0
            then t.typical_speed * least(greatest(p.pre_speed_matched / p.pre_typical_speed, 0.3), 1.3)
        else p.pre_speed
    end as expected_speed,
    if(t.typical_speed is not null and p.pre_typical_speed > 0, 'precrash_adjusted', 'precrash_only') as baseline_method
from with_typical t
left join precrash p on p.log_id = t.log_id and p.station_id = t.station_id
