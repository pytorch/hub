import argparse
import json
from collections import namedtuple

Result = namedtuple("Result", ["name", "base_time", "diff_time"])

def get_times(pytest_data):
    return {b["name"]: b["stats"]["mean"] for b in pytest_data["benchmarks"]}

parser = argparse.ArgumentParser("compare two pytest jsons")
parser.add_argument('base',  help="base json file")
parser.add_argument('diff', help='diff json file')
args = parser.parse_args()

with open(args.base, "r") as base:
    base_times = get_times(json.load(base))
with open(args.diff, "r") as diff:
    diff_times = get_times(json.load(diff))

all_keys = set(base_times.keys()).union(diff_times.keys())
results = [
    Result(name, base_times.get(name, float("nan")), diff_times.get(name, float("nan")))
    for name in sorted(all_keys)
]

print("{:48s} {:>13s} {:>15s} {:>10s}".format(
    "name", "base time (s)", "diff time (s)", "% change"))
for r in results:
    print("{:48s} {:13.6f} {:15.6f} {:9.1f}%".format(
        r.name,
        r.base_time,
        r.diff_time,
        (r.diff_time / r.base_time - 1.0) * 100.0
        ))
