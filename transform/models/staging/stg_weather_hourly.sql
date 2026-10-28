-- Hourly rainfall per 0.25-degree grid cell. Each fetch covers a Pacific day (two UTC days), so
-- neighboring days overlap: keep one row per cell and hour.
-- ts is the END of the hour; precipitation_mm is the total over the hour before it.

select
    grid_lat,
    grid_lon,
    ts,
    max(precipitation_mm) as precipitation_mm
from {{ source('bronze', 'weather_hourly') }}
group by grid_lat, grid_lon, ts
