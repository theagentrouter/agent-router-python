"""Fail release verification if required integration tests did not run."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

report = ET.parse(sys.argv[1]).getroot()
cases = report.findall(".//testcase")
if not cases or any(c.find("skipped") is not None for c in cases):
    raise SystemExit("Required integration tests were skipped or not collected")
if any(c.find("failure") is not None or c.find("error") is not None for c in cases):
    raise SystemExit("Required integration tests failed")
print(f"All {len(cases)} required tests ran and passed")
