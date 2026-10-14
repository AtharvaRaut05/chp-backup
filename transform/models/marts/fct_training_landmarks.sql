-- The training set. One row per measurable incident per landmark (10, 20 and 30 minutes after
-- CHP's log time): everything you would know at that moment, and how much longer traffic
-- stayed backed up after it.
--
-- No look-ahead: every feature only uses information available at log_time + landmark.
--   notes      only notes our poller had seen by then (first_seen_at), not CHP's note time
--   units      unit events stamped at or before the landmark
--   traffic    5-minute readings that had finished by the landmark (slot start <= landmark - 5)
--   CHP closed only if we'd detected the closure by then
--
-- Target
--   remaining_minutes  minutes from the landmark until traffic is back to normal
--                      (0 if there was no backup or it had already cleared)
--   is_censored        still backed up when we lost sight of it: remaining_minutes is a minimum
--
-- Left out: landmarks before our poller had first seen the incident (its notes would be missing
-- and some fields would come from later), and landmarks after a censored backup's last
-- observation (we don't know whether it was still going).

{% set landmarks = var('landmarks') %}

with labels as (
    select * from {{ ref('fct_incident_backup') }}
),

landmarks as (
    select l.log_id, lm.landmark_min
    from labels l
    cross join unnest(array[{{ landmarks | join(', ') }}]) as lm (landmark_min)
),

incidents as (
    select
        i.log_id,
        i.log_time,
        i.dispatch_id,
        i.district,
        i.freeway,
        i.direction,
        i.initial_type_code,
        i.closed_detected_at,
        i.first_polled_at,
        i.log_time at time zone 'America/Los_Angeles' as log_local
    from {{ ref('stg_chp_incidents') }} i
    join labels l on l.log_id = i.log_id
),

-- dispatcher notes seen by each landmark
notes as (
    select
        lm.log_id,
        lm.landmark_min,
        count(n.log_id)                                                          as n_notes,
        coalesce(bool_or(n.is_blocking), false)                                  as ever_blocking,
        -- latest note that said blocked or open decides whether lanes are blocked right now
        -- (notes first seen in the same poll are ordered by their CHP note number)
        max_by(n.is_blocking, to_unixtime(n.first_seen_at) * 10000 + coalesce(n.detail_seq, 0)) filter (where n.is_blocking or n.lanes_open) as blocking_now,
        max(n.lanes_blocked)                                                     as max_lanes_blocked,
        -- lane count in the most recent note that gave one (lanes reopen during clearance)
        max_by(n.lanes_blocked, to_unixtime(n.first_seen_at) * 10000 + coalesce(n.detail_seq, 0)) filter (where n.lanes_blocked is not null) as latest_lanes_blocked,
        coalesce(bool_or(n.all_lanes), false)                                    as all_lanes,
        coalesce(bool_or(n.lanes_open), false)                                   as lanes_reopened,
        max(n.vehicles)                                                          as vehicles,
        coalesce(bool_or(n.ambulance), false)                                    as ambulance,
        coalesce(bool_or(n.tow), false)                                          as tow,
        coalesce(bool_or(n.fsp), false)                                          as fsp,
        coalesce(bool_or(n.fire), false)                                         as fire,
        coalesce(bool_or(n.sigalert), false)                                     as sigalert,
        coalesce(bool_or(n.overturned), false)                                   as overturned,
        coalesce(bool_or(n.hazmat), false)                                       as hazmat,
        coalesce(bool_or(n.fatality), false)                                     as fatality,
        max_by(n.type_changed_to, to_unixtime(n.first_seen_at) * 10000 + coalesce(n.detail_seq, 0)) filter (where n.type_changed_to is not null) as type_changed_to
    from landmarks lm
    join incidents i on i.log_id = lm.log_id
    left join {{ ref('int_note_features') }} n
        on n.log_id = lm.log_id
       and n.first_seen_at <= i.log_time + lm.landmark_min * interval '1' minute
    group by lm.log_id, lm.landmark_min
),

-- CHP units by each landmark
units as (
    select
        lm.log_id,
        lm.landmark_min,
        count(case when u.unit_text like '%Assigned%' then 1 end)                              as units_assigned,
        count(case when u.unit_text like '%At Scene%' then 1 end)                              as units_at_scene,
        min(date_diff('minute', i.log_time, u.unit_time)) filter (where u.unit_text like '%At Scene%') as first_at_scene_min
    from landmarks lm
    join incidents i on i.log_id = lm.log_id
    left join {{ ref('stg_chp_units') }} u
        on u.log_id = lm.log_id
       and u.unit_time <= i.log_time + lm.landmark_min * interval '1' minute
    group by lm.log_id, lm.landmark_min
),

-- live traffic known at each landmark: slots that had finished by then
live as (
    select
        lm.log_id,
        lm.landmark_min,
        max_by(s.min_ratio, s.t)                                                 as queue_ratio_now,
        max_by(s.is_slowed, s.t)                                                 as backed_up_now,
        max_by(s.queue_extent_miles, s.t)                                        as extent_now_miles,
        min(s.min_ratio)                                                         as min_ratio_so_far,
        count(case when s.is_slowed then 1 end) * 5                              as slowed_minutes_so_far,
        max(s.queue_extent_miles)                                                as max_extent_so_far,
        max(s.t)                                                                 as latest_slot_min
    from landmarks lm
    left join {{ ref('int_backup_slots') }} s
        on s.log_id = lm.log_id
       and s.t <= lm.landmark_min - 5
    group by lm.log_id, lm.landmark_min
)

select
    lm.log_id,
    lm.landmark_min,
    i.log_time,

    -- when and where
    i.dispatch_id,
    i.district,
    i.freeway,
    i.direction,
    hour(i.log_local)                                                            as hour_local,
    day_of_week(i.log_local)                                                     as day_of_week,
    {{ day_type('i.log_local') }}                                                as day_type,

    -- what CHP knew
    coalesce(n.type_changed_to, i.initial_type_code)                             as type_code,
    n.n_notes,
    n.ever_blocking,
    coalesce(n.blocking_now, false)                                              as blocking_now,
    n.max_lanes_blocked,
    n.latest_lanes_blocked,
    n.all_lanes,
    n.lanes_reopened,
    n.vehicles,
    n.ambulance,
    n.tow,
    n.fsp,
    n.fire,
    n.sigalert,
    n.overturned,
    n.hazmat,
    n.fatality,
    u.units_assigned,
    u.units_at_scene,
    u.first_at_scene_min,
    coalesce(i.closed_detected_at <= i.log_time + lm.landmark_min * interval '1' minute, false) as chp_closed,

    -- before the crash
    p.site_speed,
    p.site_occupancy,
    p.site_flow,
    p.site_ratio_to_typical,
    p.site_lanes,
    p.site_speed_trend,
    p.queue_speed,
    p.queue_min_speed,
    p.queue_occupancy,
    p.queue_speed_trend,
    p.queue_sensors,
    p.approach_speed,
    p.approach_flow_per_sensor,
    p.approach_lanes,
    p.approach_flow_trend,
    p.merge_flow,
    p.connector_merge_flow,
    p.exit_flow,
    p.merge_points,

    -- normal congestion at this spot (other days' history)
    r.typical_ratio_now,
    r.typical_ratio_in_30,
    r.typical_ratio_in_60,
    r.typical_congestion_minutes_left,

    -- traffic since the crash, up to the landmark
    t.queue_ratio_now,
    coalesce(t.backed_up_now, false)                                             as backed_up_now,
    t.extent_now_miles,
    t.min_ratio_so_far,
    t.slowed_minutes_so_far,
    t.max_extent_so_far,
    t.latest_slot_min,

    -- target
    l.has_backup,
    l.backup_minutes                                                             as total_backup_minutes,
    case
        when not l.has_backup then 0
        else greatest(l.backup_start_min + l.backup_minutes - lm.landmark_min, 0)
    end                                                                          as remaining_minutes,
    l.is_censored,
    l.label_adjusted_share                          -- label quality (not a model input)

from landmarks lm
join incidents i on i.log_id = lm.log_id
join labels l on l.log_id = lm.log_id
join {{ ref('int_incident_precrash') }} p on p.log_id = lm.log_id
left join {{ ref('int_incident_recurring') }} r on r.log_id = lm.log_id
left join notes n on n.log_id = lm.log_id and n.landmark_min = lm.landmark_min
left join units u on u.log_id = lm.log_id and u.landmark_min = lm.landmark_min
left join live t on t.log_id = lm.log_id and t.landmark_min = lm.landmark_min
where i.first_polled_at <= i.log_time + lm.landmark_min * interval '1' minute
  and not (l.is_censored and l.backup_start_min + l.backup_minutes <= lm.landmark_min)
