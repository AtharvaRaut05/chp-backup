-- Typical traffic per mainline sensor, weekday vs weekend, per 5-minute slot of the day
-- (Pacific time): the normal daily pattern used to carry the pre-crash baseline forward.
-- Only measured (not imputed) readings from healthy sensor-days, and only sensors near an incident.
-- Gets more reliable as weeks of history accumulate; n_days says how many days back it.

with sensors as (
    select distinct station_id
    from {{ ref('int_incident_stations') }}
    where station_type = 'ML'
),

readings as (
    select
        r.station_id,
        r.dt,
        r.avg_speed,
        r.total_flow,
        r.ts at time zone 'America/Los_Angeles' as ts_local
    from {{ ref('stg_pems_5min') }} r
    join sensors s on s.station_id = r.station_id
    join {{ ref('int_station_day_health') }} h on h.station_id = r.station_id and h.dt = r.dt and h.is_healthy
    where r.is_observed and r.avg_speed is not null
)

select
    station_id,
    {{ day_type('ts_local') }}                               as day_type,
    {{ slot_of_day('ts_local') }}                            as slot,
    approx_percentile(avg_speed, 0.5)                        as typical_speed,
    approx_percentile(total_flow, 0.5)                       as typical_flow,
    count(distinct dt)                                       as n_days
from readings
group by 1, 2, 3
