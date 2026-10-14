-- The label: did each incident cause a backup, and how long did it last?
-- Built on int_backup_slots (the queue must start at the crash; see that model).
--
--   backup start   first slot from log_lag_minutes before the log time up to onset_minutes after
--                  it where the crash site is backed up for at least two slots in a row (10 min);
--                  a single 5-minute dip is noise. Later slowdowns aren't attributed to the incident.
--   backup end     first slot after the start where the crash site is clear (back above
--                  recover_ratio x expected) for 15 minutes straight (that slot and the next two).
--                  Slots in between "slowed" and "clear" keep the backup going.
--   is_censored    still backed up when we lose sight of it: the data window ends
--                  (series_after_minutes) or the sensors at the crash site stop reporting for
--                  longer than max_gap_minutes. The true duration is at least backup_minutes.
--   label_adjusted_share  share of the readings behind the label that had a full "typical traffic"
--                  baseline (vs. the pre-crash-speed fallback): a label-quality score, not a feature
-- Only incidents with a sensor near the crash site are labeled, and "no backup" only counts if
-- the crash site was visible steadily through the whole onset window.

with slots as (
    select * from {{ ref('int_backup_slots') }}
),

measurable as (
    select log_id from {{ ref('int_incident_precrash') }} where is_measurable
),

sequenced as (
    select
        s.*,
        lead(s.t, 1)         over (partition by s.log_id order by s.t) as t1,
        lead(s.is_slowed, 1) over (partition by s.log_id order by s.t) as slowed1,
        lead(s.is_clear, 1)  over (partition by s.log_id order by s.t) as clear1,
        lead(s.t, 2)         over (partition by s.log_id order by s.t) as t2,
        lead(s.is_clear, 2)  over (partition by s.log_id order by s.t) as clear2
    from slots s
    join measurable m on m.log_id = s.log_id
),

starts as (
    select
        log_id,
        min(t) filter (where is_slowed and slowed1 and t1 = t + 5
                         and t <= {{ var('onset_minutes') }})                     as start_t,
        min(t)                                                                   as first_t,
        max(t)                                                                   as last_t,
        coalesce(max(t1 - t) filter (where t < {{ var('onset_minutes') }}), 0)   as max_step_onset
    from sequenced
    group by log_id
),

gaps as (
    -- first slot after the start followed by a long hole (or the end of the data)
    select s.log_id, min(s.t) as cutoff_t
    from sequenced s
    join starts st on st.log_id = s.log_id
    where st.start_t is not null
      and s.t >= st.start_t
      and (s.t1 is null or s.t1 > s.t + {{ var('max_gap_minutes') }})
    group by s.log_id
),

recovery as (
    select s.log_id, min(s.t) as end_t
    from sequenced s
    join starts st on st.log_id = s.log_id
    join gaps g on g.log_id = s.log_id
    where s.t > st.start_t
      and s.t < g.cutoff_t
      and s.is_clear
      and s.t1 = s.t + 5  and s.clear1
      and s.t2 = s.t + 10 and s.clear2
    group by s.log_id
),

severity as (
    select
        s.log_id,
        greatest(max(s.queue_extent_miles), 0) as max_extent_miles,
        min(s.min_ratio)                       as min_speed_ratio,
        count_if(s.is_slowed) * 5              as slowed_minutes
    from slots s
    join starts st on st.log_id = s.log_id
    join gaps g on g.log_id = s.log_id
    left join recovery r on r.log_id = s.log_id
    where s.t >= st.start_t
      and s.t < coalesce(r.end_t, g.cutoff_t + 5)
    group by s.log_id
),

quality as (
    select log_id, avg(if(baseline_method = 'precrash_adjusted', 1.0, 0.0)) as label_adjusted_share
    from {{ ref('int_incident_traffic') }}
    where role = 'queue' and is_observed and minutes_from_log >= -{{ var('log_lag_minutes') }}
    group by log_id
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
    sv.slowed_minutes,
    q.label_adjusted_share
from starts st
left join gaps g on g.log_id = st.log_id
left join recovery r on r.log_id = st.log_id
left join severity sv on sv.log_id = st.log_id
left join quality q on q.log_id = st.log_id
where st.start_t is not null
   or (    -- "no backup" only if the crash site was visible steadily through the onset window
           st.first_t <= 5
       and st.last_t >= {{ var('onset_minutes') }}
       and st.max_step_onset <= {{ var('max_gap_minutes') }}
   )
