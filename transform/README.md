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
| `int_station_profile` | Typical speed/flow per mainline sensor, weekday/weekend, per 5-minute slot |
| `int_incident_traffic` | Traffic from 60 min before to 180 min after each incident, with the expected no-incident speed |
| `int_station_day_health` | Whether each sensor was actually measuring (not imputed) on each day |
| `int_incident_precrash` | One row per incident: conditions before the crash (site speed and trend, queue zone, approaching volume, merging volume) |
| `fct_incident_backup` | **The label**: whether each incident caused a backup, start/end, duration, censoring, how far back it reached |

**Upstream** means where traffic comes from: for a northbound or eastbound incident that's a
lower postmile (south/west of the crash); for southbound or westbound, a higher one.

**Baseline (pre-crash, adjusted):** `expected_speed = typical speed at that time of day x
(actual / typical over the 30 min before the crash)`. The pre-crash window ends 5 minutes
before CHP's log time, since crashes are logged a few minutes after they happen.

**Sensor health:** many PeMS sensors are down for whole days and PeMS fills their readings in.
A sensor only counts for an incident on days when at least 80% of its readings were measured.

**Backup label:** a queue sensor is backed up when speed < 60% of expected. The backup starts at
the first backed-up slot within 30 min of the log time and ends once no queue sensor is backed
up for 15 straight minutes. Still backed up 3 hours after the log time = censored.

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
