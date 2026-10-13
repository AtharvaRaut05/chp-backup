# Transform layer (dbt on Athena)

Reads the bronze tables written by the Lambdas (`traffic_bronze`) and builds clean tables in
`traffic_silver`.

| Model | What it is |
|---|---|
| `stg_chp_incidents` | One row per incident: latest version, freeway/direction, closure time, pairing status |
| `stg_chp_notes` | Every dispatcher note, with the time the poller first saw it |
| `stg_chp_units` | Unit events (assigned / enroute / at scene / cleared) |
| `stg_pems_stations` | Latest location, postmile and type of each sensor (mainline, ramps, connectors) |
| `stg_pems_5min` | 5-minute readings with an `is_observed` quality flag (view) |
| `stg_pems_pairing` | Current pairing status per district-day |
| `int_note_features` | Lanes blocked, vehicles, ambulance, tow, SigAlert, etc. parsed from each note |
| `int_incident_stations` | Sensors around each incident by role: `queue` (0-5 mi upstream), `approach` (5-10 mi), `merge` (on-ramps and freeway connectors feeding the queue zone), `exit` (off-ramps) |
| `int_station_profile` | Measured speed/flow per mainline sensor, per day and 5-minute slot: the history "typical traffic" is built from |
| `int_incident_traffic` | Traffic from 60 min before to 180 min after each incident, with the expected no-incident speed |
| `int_station_day_health` | Whether each sensor was actually measuring (not imputed) on each day |
| `int_incident_precrash` | One row per incident: conditions before the crash (site speed and trend, queue zone, approaching volume, merging volume) |
| `fct_incident_backup` | **The label**: whether each incident caused a backup, start/end, duration, censoring, how far back it reached |
| `fct_training_landmarks` | **The training set**: one row per incident at 10, 20 and 30 min after the log time, with everything known at that moment and the minutes left until traffic is normal |

**Upstream** means where traffic comes from: for a northbound or eastbound incident that's a
lower postmile (south/west of the crash); for southbound or westbound, a higher one.

**Baseline (pre-crash, adjusted):** `expected_speed = typical speed at that time of day x
(actual / typical over the 30 min before the crash)`. The pre-crash window ends 5 minutes
before CHP's log time, since crashes are logged a few minutes after they happen. "Typical" is
the median for that sensor and time slot on *other* days of the same type (weekday/weekend), so
the crash's own slowdown never counts as normal. With fewer than 3 such days the baseline is
just the pre-crash speed.

**Sensor health:** many PeMS sensors are down for whole days and PeMS fills their readings in.
A sensor only counts for an incident on days when at least 80% of its readings were measured.

**Backup label:** a queue sensor is backed up when speed < 60% of expected. The backup starts at
the first backed-up slot within 30 min of the log time and ends once no queue sensor is backed
up for 15 straight minutes. Still backed up when we lose sight of it (3 hours after the log time,
or the sensors stop reporting) = censored. "No backup" only counts if the sensors reported through
the whole 30-minute onset window; otherwise the incident isn't labeled.

US-101 is labeled north/south in PeMS, but CHP sometimes calls it east/west where it runs east-west
(Valley, Ventura), so CHP's E/W on 101 is mapped to S/N.

**Training set (no look-ahead):** at each landmark the model only sees notes our poller had
already seen (`first_seen_at`), unit events stamped by then, the latest incident type, CHP
closure only if detected by then, pre-crash traffic, and 5-minute readings that had finished by
then. Target: `remaining_minutes` until no queue sensor is backed up (0 if there was no backup or
it had cleared), with `is_censored` when the data ended first. Landmarks before our poller first
saw the incident, or after a censored backup's last observation, are left out.

Thresholds and windows are `vars` in `dbt_project.yml`.

## Setup (once)

```powershell
pip install dbt-athena
```

## Run

From the project root:

```powershell
$env:LAKE_BUCKET = aws cloudformation describe-stacks --stack-name chp-backup --profile traffic --query "Stacks[0].Outputs[?OutputKey=='LakeBucket'].OutputValue" --output text
cd transform
dbt debug --profiles-dir .     # checks the AWS connection
dbt seed --profiles-dir .      # loads dispatch_districts
dbt build --profiles-dir .     # builds every model and runs all tests
```

`dbt build` runs the unit tests (real CHP notes with expected features) before building
`int_note_features`, and the data tests (unique keys, valid values, relationships) after.
