-- Measured speed and flow per mainline sensor, per day and 5-minute slot (Pacific time):
-- the history that "typical traffic" is computed from in int_incident_traffic.
-- Only measured (not imputed) readings from healthy sensor-days, and only sensors near an incident.
--
-- Kept per day (not pre-averaged) so each incident's typical traffic can leave out its own day:
-- otherwise the crash's own slowdown would drag "typical" down and hide the backup.

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
    dt,
    {{ day_type('ts_local') }}    as day_type,
    {{ slot_of_day('ts_local') }} as slot,
    avg(avg_speed)                as speed,
    avg(total_flow)               as flow
from readings
group by 1, 2, 3, 4
