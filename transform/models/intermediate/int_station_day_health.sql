-- Whether each sensor was actually measuring on each day. Many PeMS sensors are down for
-- whole days (or all the time), and PeMS fills their readings in from neighbors and history.
-- Filled-in speeds don't show a backup, so a sensor only counts for an incident on days when
-- at least min_healthy_share of its readings were measured.

select
    station_id,
    lane_type,
    dt,
    avg(if(is_observed, 1.0, 0.0))                          as share_measured,
    avg(if(is_observed, 1.0, 0.0)) >= {{ var('min_healthy_share') }} as is_healthy
from {{ ref('stg_pems_5min') }}
group by station_id, lane_type, dt
