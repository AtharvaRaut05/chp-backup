-- Sensors around each freeway incident, each with a role:
--
--   queue     mainline sensors from the incident to upstream_miles upstream: where the backup builds
--   approach  mainline sensors upstream_miles to approach_miles upstream: traffic about to arrive
--   merge     on-ramps and freeway-to-freeway connectors feeding into the queue zone
--   exit      off-ramps in the queue zone: traffic that can leave before reaching the incident
--
-- "Upstream" is where traffic comes from. Caltrans postmiles increase northbound and eastbound,
-- so for a northbound (N) or eastbound (E) incident upstream means a LOWER postmile (south/west
-- of the crash), and for S/W incidents a HIGHER postmile.
--
-- The incident's postmile is that of the nearest mainline sensor on the same freeway and
-- direction. If that sensor is farther than max_match_km away, the incident is left out.

with incidents as (
    select log_id, freeway, direction, lat, lon
    from {{ ref('stg_chp_incidents') }}
    where is_traffic and freeway is not null and direction is not null and lat is not null
),

stations as (
    select station_id, type, freeway, direction, abs_pm, lat, lon, lanes, name
    from {{ ref('stg_pems_stations') }}
    where abs_pm is not null and type in ('ML', 'OR', 'FF', 'FR')
),

nearest_mainline as (
    select
        i.log_id,
        min_by(s.abs_pm, great_circle_distance(i.lat, i.lon, s.lat, s.lon)) as incident_pm,
        min(great_circle_distance(i.lat, i.lon, s.lat, s.lon))              as nearest_km
    from incidents i
    join stations s on s.freeway = i.freeway and s.direction = i.direction
    where s.type = 'ML' and s.lat is not null
    group by i.log_id
),

candidates as (
    select
        i.log_id,
        i.freeway,
        i.direction,
        n.incident_pm,
        n.nearest_km,
        s.station_id,
        s.type as station_type,
        s.name as station_name,
        s.abs_pm,
        s.lanes,
        if(i.direction in ('N', 'E'), n.incident_pm - s.abs_pm, s.abs_pm - n.incident_pm) as miles_upstream
    from incidents i
    join nearest_mainline n on n.log_id = i.log_id
    join stations s on s.freeway = i.freeway and s.direction = i.direction
    where n.nearest_km <= {{ var('max_match_km') }}
),

roles as (
    select
        *,
        case
            when station_type = 'ML' and miles_upstream between -{{ var('downstream_miles') }} and {{ var('upstream_miles') }} then 'queue'
            when station_type = 'ML' and miles_upstream > {{ var('upstream_miles') }} and miles_upstream <= {{ var('approach_miles') }} then 'approach'
            when station_type in ('OR', 'FF') and miles_upstream between 0 and {{ var('upstream_miles') }} then 'merge'
            when station_type = 'FR' and miles_upstream between 0 and {{ var('upstream_miles') }} then 'exit'
        end as role
    from candidates
)

select
    *,
    row_number() over (partition by log_id, role order by miles_upstream) as role_rank
from roles
where role is not null
