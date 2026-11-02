"""Read-only aggregate of two scalar training cohorts; this module does not train."""
from pathlib import Path
import json
import train_scalar_core as C
import train_scalar_additions as E
ROOT=C.ROOT;OUT=ROOT/'reports/toy-scalar-comparison-v1';P=json.loads((ROOT/'configs/toy-scalar-comparison-v1.json').read_text())
sha=C.sha;write=C.write;network_ids=C.network_ids;predict=E.predict

def initialize(device):
 groups,params=C.initialize(device);extra,more=E.initialize(device);groups.update(extra);return groups,params+more
