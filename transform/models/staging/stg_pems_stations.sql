-- Latest metadata for each sensor station. type: ML mainline, HV HOV, OR on-ramp,
-- FR off-ramp, FF freeway-to-freeway connector, CD collector/distributor.

select
    station_id,
    max_by(type, dt)      as type,
    max_by(district, dt)  as district,
    max_by(freeway, dt)   as freeway,
    max_by(direction, dt) as direction,
    max_by(abs_pm, dt)    as abs_pm,
    max_by(lat, dt)       as lat,
    max_by(lon, dt)       as lon,
    max_by(lanes, dt)     as lanes,
    max_by(name, dt)      as name
from {{ source('bronze', 'pems_stations') }}
group by station_id
