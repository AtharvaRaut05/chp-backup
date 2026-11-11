-- Traffic speed at the crash when CHP reported it, from the live speed lookup (TomTom).
--
--   speed_requested_at        when the lookup was made (the mart only uses it from then on)
--   live_speed_ratio          current speed / free-flow speed on the road segment (1 = flowing freely)
--   live_speed_kmh            current speed
--   live_road_closed          TomTom shows the road closed
--   live_speed_confidence     TomTom's data quality (0-1)
--
-- Only kept when the lookup landed on a freeway (road class FRC0/FRC1) heading the incident's way:
-- the segment's compass heading must be closer to the incident's direction than to the opposite one.
-- Otherwise (other carriageway, frontage road) the values are null.

with lookups as (
    select
        f.*,
        i.direction,
        f.frc in ('FRC0', 'FRC1')                                                       as on_freeway,
        cos(radians(f.segment_bearing_deg - case i.direction
            when 'N' then 0 when 'E' then 90 when 'S' then 180 when 'W' then 270 end)) > 0 as same_direction
    from {{ ref('stg_tomtom_flow') }} f
    join {{ ref('stg_chp_incidents') }} i on i.log_id = f.log_id
)

select
    log_id,
    requested_at                                                                         as speed_requested_at,
    if(on_freeway and same_direction, current_speed_kmh / nullif(free_flow_speed_kmh, 0)) as live_speed_ratio,
    if(on_freeway and same_direction, current_speed_kmh)                                 as live_speed_kmh,
    if(on_freeway and same_direction, road_closure)                                      as live_road_closed,
    if(on_freeway and same_direction, confidence)                                        as live_speed_confidence
from lookups
