-- Current pairing status per district-day (latest entry in the status log).

select
    district,
    cast(day as date)                 as day,
    max_by(status, updated_at)        as status,
    max(updated_at)                   as updated_at
from {{ source('bronze', 'pems_pairing') }}
group by district, day
