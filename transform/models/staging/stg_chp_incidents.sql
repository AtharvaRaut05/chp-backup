-- One row per incident: its latest version, how long it was open, where it is, and
-- whether that district-day has PeMS traffic data (pairing_status = 'paired').
--
-- source = 'live': seen by our poller. source = 'history': from PeMS's record of CHP incidents
-- (scripts/backfill_chp_history.py), used to fill the hours the poller has nothing for: before it
-- started, while it was down (Nov 2, 2026), and before a district was added (Bay Area, Oct 13).
-- Only in the districts the poller covers, only for district-hours with no live incident at all (so
-- an incident is never counted twice), and only for hours whose PeMS file looks complete (at least a
-- quarter of the usual number of incidents statewide for that hour of the week; some PeMS days are
-- missing most incidents).
-- History has no "first seen" or "closure detected" times; they're set to what the poller would
-- typically have recorded: seen a minute after the log time, closure noticed 3 minutes after it ended.

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
),

live as (
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
        coalesce(p.status, 'none') as pairing_status,
        'live' as source
    from located i
    left join closures c on c.log_id = i.log_id
    left join {{ ref('dispatch_districts') }} d on d.dispatch_id = i.dispatch_id
    left join pairing p
        on p.district = d.district
       and p.day = cast(date(i.log_time at time zone 'America/Los_Angeles') as date)
),

live_hours as (
    -- district-hours with at least one live incident: the poller was working there
    select distinct d.district, date_trunc('hour', l.log_time) as hour_start
    from latest l
    join {{ ref('dispatch_districts') }} d on d.dispatch_id = l.dispatch_id
),

covered_districts as (
    select distinct cast(d.district as integer) as district
    from {{ ref('dispatch_districts') }} d
    join (select distinct dispatch_id from latest) l on l.dispatch_id = d.dispatch_id
),

history_raw as (
    select
        incident_id,
        max_by(cc_code, dt)        as cc_code,
        min(log_time)              as log_time,
        max_by(log_type_code, dt)  as log_type_code,
        max_by(description, dt)    as description,
        max_by(location, dt)       as location,
        max_by(area, dt)           as area,
        max_by(lat, dt)            as lat,
        max_by(lon, dt)            as lon,
        max_by(district, dt)       as district,
        max_by(freeway, dt)        as freeway,
        max_by(direction, dt)      as direction,
        max_by(duration_min, dt)   as duration_min
    from {{ source('bronze', 'hist_chp_incidents') }}
    group by incident_id
),

history_hours as (
    select date_trunc('hour', log_time) as hour_start, count(*) as n
    from history_raw
    group by 1
),

history_typical as (
    select
        day_of_week(hour_start at time zone 'America/Los_Angeles') as dow,
        hour(hour_start at time zone 'America/Los_Angeles')        as hod,
        approx_percentile(n, 0.5)                                   as typical
    from history_hours
    group by 1, 2
),

complete_hours as (
    select h.hour_start
    from history_hours h
    join history_typical t
        on t.dow = day_of_week(h.hour_start at time zone 'America/Los_Angeles')
       and t.hod = hour(h.hour_start at time zone 'America/Los_Angeles')
    where h.n >= 0.25 * t.typical
),

history as (
    select
        'H' || h.incident_id                                                        as log_id,
        h.cc_code                                                                   as dispatch_id,
        lpad(cast(h.district as varchar), 2, '0')                                   as district,
        h.log_time,
        cast(date(h.log_time at time zone 'America/Los_Angeles') as date)           as incident_day,
        h.log_type_code                                                             as initial_type_code,
        h.log_type_code,
        h.description                                                               as log_type,
        h.log_type_code in ({{ traffic_type_codes() }})                             as is_traffic,
        h.location,
        cast(null as varchar)                                                       as location_desc,
        h.area,
        h.lat,
        h.lon,
        h.freeway,
        case
            when h.freeway = 101 and h.direction = 'E' then 'S'
            when h.freeway = 101 and h.direction = 'W' then 'N'
            else h.direction
        end                                                                         as direction,
        h.log_time + interval '1' minute                                            as first_polled_at,
        h.log_time + cast(h.duration_min as bigint) * interval '1' minute                          as last_changed_at,
        cast(1 as bigint)                                                           as n_versions,
        h.log_time + cast(h.duration_min as bigint) * interval '1' minute                          as last_seen_open,
        h.log_time + cast(h.duration_min + 3 as bigint) * interval '1' minute                    as closed_detected_at,
        h.duration_min is not null                                                  as is_closed,
        cast(h.duration_min as bigint)                                              as minutes_open,
        coalesce(p.status, 'none')                                                  as pairing_status,
        'history'                                                                   as source
    from history_raw h
    join complete_hours c on c.hour_start = date_trunc('hour', h.log_time)
    join covered_districts cd on cd.district = h.district
    left join live_hours lh
        on lh.district = lpad(cast(h.district as varchar), 2, '0')
       and lh.hour_start = date_trunc('hour', h.log_time)
    left join pairing p
        on p.district = lpad(cast(h.district as varchar), 2, '0')
       and p.day = cast(date(h.log_time at time zone 'America/Los_Angeles') as date)
    where lh.hour_start is null
      and h.duration_min >= 0
      and h.direction in ('N', 'S', 'E', 'W')
)

select * from live
union all
select * from history
