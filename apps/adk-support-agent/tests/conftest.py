# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from sdk import sdk_under_test  # noqa: E402

SDK = sdk_under_test()
os.environ["AGENTROUTER_AUTOSTART"] = "0"  # tests never start a gateway; they name the endpoint
