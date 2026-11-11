-- Calendar and daylight at each incident, all known in advance (seed: calendar_days.csv).
--
--   is_holiday          a federal or California state holiday
--   near_holiday        the day before or after one
--   long_weekend        part of a 3- or 4-day holiday weekend (Thanksgiving Thu-Sun)
--   getaway_day         the last workday before a long weekend (heavy afternoon traffic out of town)
--   winter_break        Dec 21 - Jan 3: schools out, many offices quiet
--   dst_week            the day the clocks change and the week after (the commute shifts into darkness)
--   special_day         Halloween, Christmas Eve, New Year's Eve
--   sun_elevation_deg   height of the sun above the horizon at the crash (negative = below)
--   is_dark             sun more than 6 degrees below the horizon (after civil twilight)
--   sun_glare           low sun (0-15 degrees up) within 30 degrees of the direction of travel
--
-- Sun position: the standard low-precision formulas (Astronomical Almanac), good to about a degree.
-- Direction of travel: the road's heading from OpenStreetMap where the incident was placed on the
-- road, else the freeway direction (N = 0, E = 90, S = 180, W = 270 degrees).

with incidents as (
    select
        i.log_id,
        i.incident_day,
        i.lat,
        i.lon,
        coalesce(r.road_heading_deg, case i.direction
            when 'N' then 0 when 'E' then 90 when 'S' then 180 when 'W' then 270 end) as heading_deg,
        to_unixtime(i.log_time) / 86400.0 - 10957.5                                  as d   -- days since J2000
    from {{ ref('stg_chp_incidents') }} i
    left join {{ ref('stg_road_layout') }} r on r.log_id = i.log_id
    where i.is_traffic and i.lat is not null
),

ecliptic as (
    select
        *,
        radians(357.529 + 0.98560028 * d)                                            as g,     -- mean anomaly
        radians(23.439 - 0.00000036 * d)                                             as obl,   -- obliquity
        radians(mod(18.697374558 + 24.06570982441908 * d, 24) * 15 + lon)            as lst    -- local sidereal time
    from incidents
),

equatorial as (
    select
        *,
        radians(280.459 + 0.98564736 * d + 1.915 * sin(g) + 0.020 * sin(2 * g))     as ecl_lon
    from ecliptic
),

sun as (
    select
        *,
        asin(sin(obl) * sin(ecl_lon))                                                 as declination,
        lst - atan2(cos(obl) * sin(ecl_lon), cos(ecl_lon))                            as hour_angle
    from equatorial
),

sun_position as (
    select
        log_id,
        incident_day,
        heading_deg,
        degrees(asin(sin(radians(lat)) * sin(declination) + cos(radians(lat)) * cos(declination) * cos(hour_angle))) as elevation,
        mod(degrees(atan2(sin(hour_angle),
            cos(hour_angle) * sin(radians(lat)) - tan(declination) * cos(radians(lat)))) + 540, 360)         as azimuth
    from sun
),

days as (
    select
        day,
        bool_or(kind = 'holiday')       as is_holiday,
        bool_or(kind = 'long_weekend')  as long_weekend,
        bool_or(kind = 'getaway_day')   as getaway_day,
        bool_or(kind = 'winter_break')  as winter_break,
        bool_or(kind = 'dst_week')      as dst_week,
        bool_or(kind = 'special_day')   as special_day
    from {{ ref('calendar_days') }}
    group by day
)

select
    p.log_id,
    coalesce(d.is_holiday, false)                                                                 as is_holiday,
    coalesce(tomorrow.is_holiday, false) or coalesce(yesterday.is_holiday, false)                       as near_holiday,
    coalesce(d.long_weekend, false)                                                               as long_weekend,
    coalesce(d.getaway_day, false)                                                                as getaway_day,
    coalesce(d.winter_break, false)                                                               as winter_break,
    coalesce(d.dst_week, false)                                                                   as dst_week,
    coalesce(d.special_day, false)                                                                as special_day,
    round(p.elevation, 1)                                                                         as sun_elevation_deg,
    p.elevation < -6                                                                              as is_dark,
    p.elevation between 0 and 15
        and abs(mod(p.azimuth - p.heading_deg + 540, 360) - 180) < 30                              as sun_glare
from sun_position p
left join days d on d.day = p.incident_day
left join days tomorrow on tomorrow.day = p.incident_day + interval '1' day
left join days yesterday on yesterday.day = p.incident_day - interval '1' day
