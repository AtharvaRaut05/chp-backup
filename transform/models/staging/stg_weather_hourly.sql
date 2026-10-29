-- Hourly rainfall per 0.25-degree grid cell. ts is the END of the hour; precipitation_mm is the
-- total over the hour before it. Neighboring days' fetches overlap and a later run can fetch the
-- same hour again: keep the newest download of each cell and hour.

select
    grid_lat,
    grid_lon,
    ts,
    max_by(precipitation_mm, coalesce(fetched_at, timestamp '1970-01-01 00:00:00')) as precipitation_mm
from {{ source('bronze', 'weather_hourly') }}
group by grid_lat, grid_lon, ts
