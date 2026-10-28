-- Rain at each incident's location before CHP logged it (Open-Meteo, 0.25-degree grid).
--
--   rain_last_hour_mm   rain in the last full hour before the log time
--   rain_last_3h_mm     ... in the last 3 full hours
--   rain_last_24h_mm    ... in the last 24 full hours (wet roads, and the first rain after a dry spell)
--
-- Only hours that had ended by the log time count, so this is known when the incident is logged.
-- Incidents without weather for their cell and day get nulls, not zeros.

with incidents as (
    select
        log_id,
        round(lat * 4) / 4                 as grid_lat,
        round(lon * 4) / 4                 as grid_lon,
        date_trunc('hour', log_time)       as last_hour_end
    from {{ ref('stg_chp_incidents') }}
    where lat is not null and lon is not null
)

select
    i.log_id,
    sum(w.precipitation_mm) filter (where w.ts = i.last_hour_end)                                    as rain_last_hour_mm,
    sum(w.precipitation_mm) filter (where w.ts > i.last_hour_end - interval '3' hour)                as rain_last_3h_mm,
    sum(w.precipitation_mm)                                                                          as rain_last_24h_mm
from incidents i
join {{ ref('stg_weather_hourly') }} w
    on w.grid_lat = i.grid_lat
   and w.grid_lon = i.grid_lon
   and w.ts <= i.last_hour_end
   and w.ts > i.last_hour_end - interval '24' hour
group by i.log_id
