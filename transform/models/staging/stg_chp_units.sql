-- Unit events (assigned / enroute / at scene / cleared) from each incident's latest version.
-- Several units can log the same event at the same minute, so rows are not deduplicated.

with latest_version as (
    select log_id, max(polled_at) as polled_at
    from {{ source('bronze', 'chp_incidents') }}
    group by log_id
)

select u.log_id, u.unit_seq, u.unit_time, u.unit_text
from {{ source('bronze', 'chp_units') }} u
join latest_version v on v.log_id = u.log_id and v.polled_at = u.polled_at
