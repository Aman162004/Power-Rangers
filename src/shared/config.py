"""Environment configuration and settings."""

import os

# Fallback data configuration
# When True, the backend returns dummy data sampled from historical data
# when model inference fails. When False, inference failures return HTTP 500.
ENABLE_DUMMY_FALLBACK = os.getenv("ENABLE_DUMMY_FALLBACK", "false").lower() == "true"
