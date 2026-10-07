-- Every dispatcher note per incident, taken from the incident's latest version (notes only
-- accumulate), with the time our poller first saw it. Use first_seen_at, not detail_time,
-- when building features "known at time t" so the model never sees the future.

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
)

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
