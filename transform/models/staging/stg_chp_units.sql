-- Unit events (assigned / enroute / at scene / cleared) from each incident's latest version.
-- Several units can log the same event at the same minute, so rows are not deduplicated.
-- History incidents (see stg_chp_incidents) record unit events as notes; they're taken from there.

with latest_version as (
    select log_id, max(polled_at) as polled_at
    from {{ source('bronze', 'chp_incidents') }}
    group by log_id
)

select u.log_id, u.unit_seq, u.unit_time, u.unit_text
from {{ source('bronze', 'chp_units') }} u
join latest_version v on v.log_id = u.log_id and v.polled_at = u.polled_at

union all

select i.log_id, cast(null as bigint) as unit_seq, d.detail_time as unit_time, d.detail_text as unit_text
from {{ source('bronze', 'hist_chp_details') }} d
join {{ ref('stg_chp_incidents') }} i on i.log_id = 'H' || d.incident_id and i.source = 'history'
where d.detail_text in ('Unit Assigned', 'Unit Enroute', 'Unit At Scene', 'Unit Cleared')
