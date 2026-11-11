"""Save Caltrans' lane closure feed every 5 minutes.

The Lane Closure System lists every approved lane, ramp and road closure for the next 7 days plus
the ones in place now, with the lanes closed and when crews actually started (code 1097) and
finished (1098). A crash inside a work zone, where lanes are already closed, backs up faster.

The feed only shows current and upcoming closures, so history exists only from the day this starts.
Files are saved raw under raw/caltrans_lcs/; jobs/lane_closures.py turns each day's files into the
bronze table caltrans_lane_closures. Districts match the CHP centers collected (7, 8, 11, 12).
Fields: https://cwwp2.dot.ca.gov/documentation/lcs/lcs-field-description.htm
"""

from caltrans.cwwp import save_districts

DISTRICTS = (7, 8, 11, 12)
URL = "https://cwwp2.dot.ca.gov/data/d{d}/lcs/lcsStatusD{d:02d}.json"


def main(event, context):
    return save_districts("caltrans_lcs", URL, DISTRICTS)
