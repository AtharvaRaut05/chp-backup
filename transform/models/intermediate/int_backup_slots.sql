-- Backup state of each incident in each 5-minute slot, from the queue-zone sensors (the 5 miles
-- upstream of the crash). Shared by the label (fct_incident_backup) and the live-traffic features
-- (fct_training_landmarks), so both use exactly the same definition.
--
-- A sensor reading is
--   slowed   speed < slow_ratio x expected speed, or no speed but the sensor is at least
--            jam_occupancy full (stopped traffic often has no measurable speed)
--   clear    speed >= recover_ratio x expected speed
--   neither  in between (still recovering)
--
-- A slot is
--   is_slowed  a sensor within head_miles of the crash is slowed: the queue has to start AT the
--              crash. Slow sensors farther back with normal traffic in between are a different
--              bottleneck and don't count.
--   is_clear   every reporting sensor within head_miles of the crash is clear
--   queue_extent_miles  how far back the queue reaches: the unbroken run of slowed sensors
--              starting from the first slowed sensor near the crash
-- Slots with no reporting sensor within head_miles of the crash are left out (we can't see the
-- crash site), which the label treats as a gap in the data.

{% set head = var('head_miles') %}

with readings as (
    select
        log_id,
        minutes_from_log as t,
        station_id,
        miles_upstream,
        case when avg_speed is not null and expected_speed > 0 then avg_speed / expected_speed end as ratio,
        avg_occupancy
    from {{ ref('int_incident_traffic') }}
    where role = 'queue'
      and is_observed
      and minutes_from_log >= -{{ var('log_lag_minutes') }}
),

states as (
    select
        *,
        coalesce(ratio < {{ var('slow_ratio') }}, false)
          or (ratio is null and coalesce(avg_occupancy >= {{ var('jam_occupancy') }}, false)) as slowed,
        coalesce(ratio >= {{ var('recover_ratio') }}, false)                                  as clear,
        miles_upstream <= {{ head }}                                                           as near_crash,
        row_number() over (partition by log_id, t order by miles_upstream)                    as rn
    from readings
    where ratio is not null or avg_occupancy is not null
),

head as (
    select
        log_id,
        t,
        min(rn) filter (where slowed and near_crash) as head_rn,
        count(*) filter (where near_crash)           as near_sensors,
        bool_and(clear) filter (where near_crash)    as near_all_clear,
        min(ratio) filter (where near_crash)         as near_min_ratio
    from states
    group by log_id, t
),

run_end as (
    -- the first sensor past the head that isn't slowed ends the queue
    select s.log_id, s.t, min(s.rn) filter (where not s.slowed and s.rn > h.head_rn) as break_rn
    from states s
    join head h on h.log_id = s.log_id and h.t = s.t
    where h.head_rn is not null
    group by s.log_id, s.t
),

queue as (
    select s.log_id, s.t, max(s.miles_upstream) as queue_extent_miles, min(s.ratio) as queue_min_ratio
    from states s
    join head h on h.log_id = s.log_id and h.t = s.t
    join run_end r on r.log_id = s.log_id and r.t = s.t
    where s.rn >= h.head_rn and (r.break_rn is null or s.rn < r.break_rn)
    group by s.log_id, s.t
)

select
    h.log_id,
    h.t,
    h.head_rn is not null                                    as is_slowed,
    h.head_rn is null and coalesce(h.near_all_clear, false)  as is_clear,
    greatest(q.queue_extent_miles, 0)                        as queue_extent_miles,
    coalesce(q.queue_min_ratio, h.near_min_ratio)            as min_ratio
from head h
left join queue q on q.log_id = h.log_id and q.t = h.t
where h.near_sensors > 0
