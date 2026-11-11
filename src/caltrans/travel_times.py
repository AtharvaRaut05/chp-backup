"""Save Caltrans' real-time travel-time files every 5 minutes.

Caltrans publishes the current travel time on selected freeway segments, refreshed every 5 minutes,
for districts 3, 8, 11 and 12 (https://cwwp2.dot.ca.gov/documentation/tt/tt.htm). PeMS gives this
project traffic a day late; these files are available within minutes, so with enough history they
can stand in for "traffic right before the crash" in a model that runs live.

Files are saved raw (gzipped, one per district per run) under raw/caltrans_tt/. They are parsed once
there's enough history to model with, so nothing about their layout is assumed here beyond JSON.
"""

from caltrans.cwwp import save_districts

DISTRICTS = (8, 11, 12, 3)
URL = "https://cwwp2.dot.ca.gov/data/d{d}/tt/ttStatusD{d:02d}.json"


def main(event, context):
    return save_districts("caltrans_tt", URL, DISTRICTS)
