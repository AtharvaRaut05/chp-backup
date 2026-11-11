-- Every dispatcher note per incident, taken from the incident's latest version (notes only
-- accumulate), with the time our poller first saw it. Use first_seen_at, not detail_time,
-- when building features "known at time t" so the model never sees the future.
--
-- History incidents (PeMS's record, see stg_chp_incidents) have no "first seen" time. Compared with
-- what the poller saw on the weeks both exist, the history's time (the later of CHP's stamp and any
-- "[Appended hh:mm:ss]" posting time) runs a median 1.8 minutes early and within 4 minutes for 95% of
-- notes, so first_seen_at = that time + 3 minutes. Unit events are left to stg_chp_units.

with notes as (
    select * from {{ source('bronze', 'chp_details') }}
),

latest_version as (
    select log_id, max(polled_at) as polled_at
    from {{ source('bronze', 'chp_incidents') }}
    group by log_id
),

first_seen as (
    select log_id, detail_seq, detail_text, min(polled_at) as first_seen_at
    from notes
    group by log_id, detail_seq, detail_text
),

live as (
    select
        n.log_id,
        n.detail_seq,
        n.detail_time,
        f.first_seen_at,
        n.detail_text
    from notes n
    join latest_version v on v.log_id = n.log_id and v.polled_at = n.polled_at
    join first_seen f
        on f.log_id = n.log_id
       and f.detail_text = n.detail_text
       and coalesce(f.detail_seq, -1) = coalesce(n.detail_seq, -1)
),

history as (
    select
        i.log_id,
        try_cast(regexp_extract(d.detail_text, '^\s*\[(\d+)\]', 1) as bigint)          as detail_seq,
        d.detail_time,
        d.seen_at + interval '3' minute                                                     as first_seen_at,
        -- same cleanup as the live feed: drop "[n]" / "[Appended ...]" prefixes and "[Shared]" suffixes
        trim(regexp_replace(regexp_replace(regexp_replace(d.detail_text,
            '^\s*(\[(\d+|Appended[ ,]+\d{2}:\d{2}:\d{2})\]\s*)+', ''),
            '(\s*\[Shared\])+\s*$', ''), '\s+', ' '))                                     as detail_text
    from {{ source('bronze', 'hist_chp_details') }} d
    join {{ ref('stg_chp_incidents') }} i on i.log_id = 'H' || d.incident_id and i.source = 'history'
    where d.detail_text not in ('Unit Assigned', 'Unit Enroute', 'Unit At Scene', 'Unit Cleared')
)

select * from live
union all
select * from history
