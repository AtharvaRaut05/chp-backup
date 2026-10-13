-- The label: did each incident cause a backup, and how long did it last?
--
-- A queue sensor is "backed up" in a 5-minute slot when its measured speed is below
-- slow_ratio x expected speed (the pre-crash-adjusted baseline).
--   backup start   first backed-up slot from log_lag_minutes before the log time up to
--                  onset_minutes after it (later slowdowns aren't attributed to the incident)
--   backup end     first slot after the start where no queue sensor is backed up for
--                  15 minutes straight (that slot and the next two)
--   is_censored    still backed up when we lose sight of it: the data window ends
--                  (series_after_minutes) or the queue sensors stop reporting. The true
--                  duration is at least backup_minutes, as in survival analysis.
-- Only measurable incidents (a healthy queue sensor before the crash) are labeled, and
-- "no backup" only counts if the sensors reported steadily through the whole onset window.
-- Holes of up to max_gap_minutes in the readings are tolerated; a longer one ends what we can see.

{% set ratio = var('slow_ratio') %}

with readings as (
    select
        log_id,
        minutes_from_log,
        miles_upstream,
        avg_speed / expected_speed as speed_ratio
    from {{ ref('int_incident_traffic') }}
    where role = 'queue'
      and is_observed
      and avg_speed is not null
      and expected_speed > 0
      and minutes_from_log >= -{{ var('log_lag_minutes') }}
),

slots as (
    select
        log_id,
        minutes_from_log                                    as t,
        min(speed_ratio)                                    as min_ratio,
        bool_or(speed_ratio < {{ ratio }})                  as is_slowed,
        max(case when speed_ratio < {{ ratio }} then miles_upstream end) as slowed_extent
    from readings
    group by log_id, minutes_from_log
),

measurable as (
    select log_id from {{ ref('int_incident_precrash') }} where is_measurable
),

steps as (
    select *, lead(t) over (partition by log_id order by t) - t as step
    from slots
),

starts as (
    select
        s.log_id,
        min(s.t) filter (where s.is_slowed and s.t <= {{ var('onset_minutes') }}) as start_t,
        min(s.t)                                                               as first_t,
        max(s.t)                                                               as last_t,
        coalesce(max(s.step) filter (where s.t < {{ var('onset_minutes') }}), 0) as max_step_onset
    from steps s
    join measurable m on m.log_id = s.log_id
    group by s.log_id
),

sequenced as (
    select
        s.*,
        st.start_t,
        lead(s.t, 1)         over (partition by s.log_id order by s.t) as t1,
        lead(s.is_slowed, 1) over (partition by s.log_id order by s.t) as slowed1,
        lead(s.t, 2)         over (partition by s.log_id order by s.t) as t2,
        lead(s.is_slowed, 2) over (partition by s.log_id order by s.t) as slowed2
    from slots s
    join starts st on st.log_id = s.log_id
),

-- first slot after the start followed by a long hole (or the end of the data): the backup is
-- observed up to here. Short holes (a missed reading or two) are tolerated.
gaps as (
    select log_id, min(t) as cutoff_t
    from sequenced
    where start_t is not null
      and t >= start_t
      and (t1 is null or t1 > t + {{ var('max_gap_minutes') }})
    group by log_id
),

recovery as (
    select s.log_id, min(s.t) as end_t
    from sequenced s
    join gaps g on g.log_id = s.log_id
    where s.t > s.start_t
      and s.t < g.cutoff_t
      and not s.is_slowed
      and s.t1 = s.t + 5 and not s.slowed1
      and s.t2 = s.t + 10 and not s.slowed2
    group by s.log_id
),

severity as (
    select
        s.log_id,
        greatest(max(s.slowed_extent), 0) as max_extent_miles,
        min(s.min_ratio)             as min_speed_ratio,
        count_if(s.is_slowed) * 5    as slowed_minutes
    from slots s
    join starts st on st.log_id = s.log_id
    join gaps g on g.log_id = s.log_id
    left join recovery r on r.log_id = s.log_id
    where s.t >= st.start_t
      and s.t < coalesce(r.end_t, g.cutoff_t + 5)
    group by s.log_id
)

select
    st.log_id,
    st.start_t is not null                            as has_backup,
    st.start_t                                        as backup_start_min,
    r.end_t                                           as backup_end_min,
    case
        when st.start_t is null then 0
        when r.end_t is not null then r.end_t - st.start_t
        else g.cutoff_t + 5 - st.start_t
    end                                               as backup_minutes,
    st.start_t is not null and r.end_t is null        as is_censored,
    coalesce(g.cutoff_t, st.last_t) + 5               as observed_until_min,
    sv.max_extent_miles,
    sv.min_speed_ratio,
    sv.slowed_minutes
from starts st
left join gaps g on g.log_id = st.log_id
left join recovery r on r.log_id = st.log_id
left join severity sv on sv.log_id = st.log_id
where st.start_t is not null
   or (    -- "no backup" only if the sensors reported steadily through the onset window
           st.first_t <= 5
       and st.last_t >= {{ var('onset_minutes') }}
       and st.max_step_onset <= {{ var('max_gap_minutes') }}
   )
