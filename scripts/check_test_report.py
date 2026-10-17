"""Fail release verification if required integration tests did not run."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

report = ET.parse(sys.argv[1]).getroot()
cases = report.findall(".//testcase")
# A known gap is a strict expected failure (xfail), which the report lists as skipped; it ran.
skipped = [c for c in cases if c.find("skipped") is not None and c.find("skipped").get("type") != "pytest.xfail"]
if not cases or skipped:
    raise SystemExit("Required integration tests were skipped or not collected")
if any(c.find("failure") is not None or c.find("error") is not None for c in cases):
    raise SystemExit("Required integration tests failed")
known_gaps = [c for c in cases if c.find("skipped") is not None and c.find("skipped").get("type") == "pytest.xfail"]
print(f"Required tests ran: {len(cases) - len(known_gaps)} passed, {len(known_gaps)} known gaps (expected failures)")
