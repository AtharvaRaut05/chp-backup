-- One row per incident: its latest version, how long it was open, where it is, and
-- whether that district-day has PeMS traffic data (pairing_status = 'paired').

with versions as (
    select * from {{ source('bronze', 'chp_incidents') }}
),

latest as (
    select
        log_id,
        max_by(dispatch_id, polled_at)    as dispatch_id,
        min(log_time)                     as log_time,
        min_by(log_type_code, polled_at)  as initial_type_code,
        max_by(log_type_code, polled_at)  as log_type_code,
        max_by(log_type, polled_at)       as log_type,
        max_by(location, polled_at)       as location,
        max_by(location_desc, polled_at)  as location_desc,
        max_by(area, polled_at)           as area,
        max_by(lat, polled_at)            as lat,
        max_by(lon, polled_at)            as lon,
        min(polled_at)                    as first_polled_at,
        max(polled_at)                    as last_changed_at,
        count(*)                          as n_versions
    from versions
    group by log_id
),

closures as (
    -- A closure only counts if the incident never showed up again afterwards (a feed outage
    -- or an overlapping poll can record a closure for an incident that's still open).
    select c.log_id, max(c.last_seen) as last_seen_open, max(c.closed_detected_at) as closed_detected_at
    from {{ source('bronze', 'chp_closures') }} c
    join latest l on l.log_id = c.log_id
    where c.closed_detected_at > l.last_changed_at
    group by c.log_id
),

located as (
    select
        l.*,
        -- "I5 N / Main St" or "SR91 W" in the location (or its description); else "SB 405 JNO ..."
        coalesce(
            regexp_extract(upper(location), '\b(?:I|SR|US)([0-9]{1,3}) [NSEW]\b', 1),
            regexp_extract(upper(location_desc), '\b(?:I|SR|US)([0-9]{1,3}) [NSEW]\b', 1),
            regexp_extract(upper(location_desc), '\b[NSEW]B ([0-9]{1,3})\b', 1)
        ) as freeway_txt,
        coalesce(
            regexp_extract(upper(location), '\b(?:I|SR|US)[0-9]{1,3} ([NSEW])\b', 1),
            regexp_extract(upper(location_desc), '\b(?:I|SR|US)[0-9]{1,3} ([NSEW])\b', 1),
            regexp_extract(upper(location_desc), '\b([NSEW])B [0-9]{1,3}\b', 1)
        ) as direction_txt
    from latest l
),

pairing as (
    select * from {{ ref('stg_pems_pairing') }}
)

select
    i.log_id,
    i.dispatch_id,
    d.district,
    i.log_time,
    cast(date(i.log_time at time zone 'America/Los_Angeles') as date) as incident_day,
    i.initial_type_code,
    i.log_type_code,
    i.log_type,
    i.log_type_code in ({{ traffic_type_codes() }}) or i.log_type = 'SIG Alert' as is_traffic,
    i.location,
    i.location_desc,
    i.area,
    i.lat,
    i.lon,
    cast(i.freeway_txt as integer) as freeway,
    -- PeMS labels US-101 north/south everywhere; CHP sometimes says east/west where 101 runs
    -- east-west (San Fernando Valley, Ventura). Northbound 101 heads west there.
    case
        when i.freeway_txt = '101' and i.direction_txt = 'E' then 'S'
        when i.freeway_txt = '101' and i.direction_txt = 'W' then 'N'
        else i.direction_txt
    end as direction,
    i.first_polled_at,
    i.last_changed_at,
    i.n_versions,
    c.last_seen_open,
    c.closed_detected_at,
    c.closed_detected_at is not null as is_closed,
    date_diff('minute', i.log_time, c.last_seen_open) as minutes_open,
    coalesce(p.status, 'none') as pairing_status
from located i
left join closures c on c.log_id = i.log_id
left join {{ ref('dispatch_districts') }} d on d.dispatch_id = i.dispatch_id
left join pairing p
    on p.district = d.district
   and p.day = cast(date(i.log_time at time zone 'America/Los_Angeles') as date)
