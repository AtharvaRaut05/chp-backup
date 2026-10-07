-- Structured features pulled from each dispatcher note with regular expressions.
-- Patterns were tested against real CHP notes; see the unit tests in intermediate.yml.
--
--   is_blocking     note says a lane is blocked ("BLKG LN 2", "1125 IN #1 LN", "HOLDING ALL LNS")
--   lanes_blocked   how many lanes, when the note says ("LN 2,3,4" -> 3, "#3/4" -> 2, "2 RT LNS" -> 2)
--   all_lanes       all or both lanes blocked
--   lanes_open      lanes reopened / roadway clear
--   vehicles        vehicles involved ("3 VEHS", "SOLO", "A VS B VS C"; fixed objects not counted)
--   type_changed_to new incident code from "Problem changed from X to Y" notes

with notes as (
    select log_id, detail_seq, detail_time, first_seen_at, detail_text, upper(detail_text) as t
    from {{ ref('stg_chp_notes') }}
),

matched as (
    select
        *,
        regexp_like(t, '\b(?:BLK''?G|BLGK|BLKD|BLOCKING|BLOCKED|BLOCK)\b|\bLNS?\b[^/]{0,15}\bCLOSED\b|\b1125 (?:IN |SB |NB |EB |WB )?(?:THE )?(?:#|LNS?\b|LANES?\b|ALL\b|BOTH\b|MIDDLE\b|SLOW\b|FAST\b|HOV\b)|\bIN (?:THE )?#[1-8]|\bSTALLED IN LN') as says_blocked,
        regexp_like(t, '\bNEG (?:BLK''?G|BLOCKING|1125)\b') as says_not_blocked,
        regexp_like(t, '\b(?:LNS?|LANES?)\b[^/]{0,15}\b(?:OPEN|CLR|CLEAR)\b|\bRDWY (?:IS )?CLR\b|\bRDWY CLEAR|\bROADWAY CLEAR') as lanes_open,
        regexp_like(t, '\b(?:ALL|BOTH) (?:LNS|LANES|LN)\b') as says_all_lanes,
        coalesce(
            regexp_extract(t, '\b(?:LNS?|LANES?) ?#?([1-8](?![0-9])(?: ?(?:,|-|/|&|AND) ?#?[1-8](?![0-9]))*)', 1),
            regexp_extract(t, '#([1-8](?![0-9])(?: ?(?:,|-|/|&|AND) ?#?[1-8](?![0-9]))*)', 1),
            regexp_extract(t, '\bBLKG ([1-8](?![0-9])(?: ?(?:,|-|/|&|AND) ?[1-8](?![0-9]))*)\b', 1)
        ) as lane_spec,
        try_cast(regexp_extract(t, '\b([1-8]) (?:RT |RIGHT |LT |LEFT |MID |MIDDLE |CTR |CENTER |SLOW |FAST )?LNS\b', 1) as integer) as n_lanes_before,
        regexp_like(t, '\b(?:SLOW|SLO|FAST|MIDDLE|MID|CTR|CENTER|HOV|CARPOOL|LEFT|RIGHT|RT|LT) LN\b') as named_single_lane,
        try_cast(regexp_extract(t, '(?<![0-9-])([1-9])(?:-[1-9])? ?VEHS?\b', 1) as integer) as veh_explicit,
        regexp_like(t, '\bSOLO\b|\bONE VEH\b') as solo,
        cardinality(regexp_extract_all(t, ' VS\.? ')) as n_vs,
        cardinality(regexp_extract_all(t, ' VS\.? (?:THE )?(?:POLE|CD|GR|GUARDRAIL|TREE|CURB|WALL|BRDG|BRIDGE|FENCE|SIGN|DEBRIS|K-?RAIL|ATTENUATOR|DITCH|MAILBOX|POST|BARRIER|DEER|DOG|COW|HORSE|ANIMAL|ORCHARD|BUILDING|HOUSE|PED)\b')) as n_vs_object
    from notes
),

flags as (
    select
        *,
        says_all_lanes and not lanes_open                                        as all_lanes,
        (says_blocked or (says_all_lanes and not lanes_open)) and not says_not_blocked as is_blocking
    from matched
)

select
    log_id,
    detail_seq,
    detail_time,
    first_seen_at,
    detail_text,
    is_blocking,
    all_lanes,
    lanes_open,
    case
        when not is_blocking or all_lanes then null
        when lane_spec is not null then cardinality(array_distinct(flatten(transform(
            regexp_extract_all(lane_spec, '[1-8] ?- ?[1-8]|[1-8]'),
            tok -> if(
                strpos(tok, '-') > 0,
                sequence(
                    least(cast(substr(tok, 1, 1) as bigint), cast(substr(tok, length(tok), 1) as bigint)),
                    greatest(cast(substr(tok, 1, 1) as bigint), cast(substr(tok, length(tok), 1) as bigint))
                ),
                array[cast(tok as bigint)]
            )
        ))))
        when n_lanes_before is not null then n_lanes_before
        when named_single_lane then 1
    end                                                                         as lanes_blocked,
    coalesce(veh_explicit, if(solo, 1), if(n_vs > 0, 1 + n_vs - n_vs_object))  as vehicles,
    regexp_like(t, '\b1141\b|AMBULANCE|\bAMB\b|PARAMEDIC|\bEMS\b')             as ambulance,
    regexp_like(t, '\b1185\b|\bTOW\b|TOWING|WRECKER')                           as tow,
    regexp_like(t, '\bFSP\b')                                                   as fsp,
    regexp_like(t, '\bFIRE\b|\bFD\b|CAL ?FIRE|\bCFD\b|\bENGINE\b')              as fire,
    regexp_like(t, 'SIG ?ALERT')                                                as sigalert,
    regexp_like(t, '\bOT\b|OVERTURN|ROLL ?OVER|ROLLED|ON (?:ITS )?ROOF|\bON (?:ITS )?SIDE\b') as overturned,
    regexp_like(t, 'HAZ ?MAT|SPILL|LEAK(?:ING)? (?:FUEL|GAS|DIESEL|OIL)')       as hazmat,
    regexp_like(t, '\b1144\b')                                                  as fatality,
    regexp_extract(t, 'PROBLEM CHANGED FROM .*? TO ([0-9A-Z]+)-', 1)            as type_changed_to
from flags
