-- Normal (recurring) congestion at each incident's site sensor, from OTHER days' history: how
-- slow this spot usually is right now and over the next hour, and how long until it usually
-- flows freely again. All of it is known before the crash, so it's safe as a model input.
--
-- A crash at 5:15 PM where rush hour normally clears at 7 behaves differently from one at
-- 6:50 PM; these columns tell the model which situation it's in.
--
--   free_flow_speed                  the sensor's typical speed when traffic flows freely
--                                    (90th percentile of its typical speeds across the day)
--   typical_ratio_now / _in_30 / _in_60
--                                    typical speed now, 30 and 60 min later, / free-flow speed
--   typical_congestion_minutes_left  minutes until the typical speed reaches 90% of free flow
--                                    (0 = normally free-flowing now; 180 = not within 3 hours)
-- Uses the same weekday/weekend history as the label's baseline, leaving out the incident's
-- own day, and only time slots with at least min_profile_days of history.

with incidents as (
    select
        i.log_id,
        cast(i.incident_day as varchar)                    as incident_dt,
        p.site_station_id,
        i.log_time at time zone 'America/Los_Angeles'      as log_local
    from {{ ref('stg_chp_incidents') }} i
    join {{ ref('int_incident_precrash') }} p on p.log_id = i.log_id
    where p.site_station_id is not null
),

keyed as (
    select
        log_id,
        incident_dt,
        site_station_id,
        {{ day_type('log_local') }}    as day_type,
        {{ slot_of_day('log_local') }} as log_slot
    from incidents
),

-- typical speed per time slot at the site sensor, on other days of the same type
history as (
    select
        k.log_id,
        k.log_slot,
        p.slot,
        approx_percentile(p.speed, 0.5) as typical_speed
    from keyed k
    join {{ ref('int_station_profile') }} p
        on p.station_id = k.site_station_id
       and p.day_type = k.day_type
       and p.dt <> k.incident_dt
    group by k.log_id, k.log_slot, p.slot
    having count(*) >= {{ var('min_profile_days') }}
),

free_flow as (
    select log_id, approx_percentile(typical_speed, 0.9) as free_flow_speed
    from history
    group by log_id
),

ahead as (
    select
        h.log_id,
        (h.slot - h.log_slot) * 5          as minutes_ahead,
        h.typical_speed / f.free_flow_speed as typical_ratio
    from history h
    join free_flow f on f.log_id = h.log_id
    where h.slot between h.log_slot and h.log_slot + 36   -- the next 3 hours (same day)
      and f.free_flow_speed > 0
)

select
    a.log_id,
    f.free_flow_speed,
    max(a.typical_ratio) filter (where a.minutes_ahead = 0)                  as typical_ratio_now,
    max(a.typical_ratio) filter (where a.minutes_ahead = 30)                 as typical_ratio_in_30,
    max(a.typical_ratio) filter (where a.minutes_ahead = 60)                 as typical_ratio_in_60,
    coalesce(min(a.minutes_ahead) filter (where a.typical_ratio >= 0.9), 180) as typical_congestion_minutes_left
from ahead a
join free_flow f on f.log_id = a.log_id
group by a.log_id, f.free_flow_speed
