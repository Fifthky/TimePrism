from gluonts.dataset.repository import get_dataset
import gdown
import os
import sys
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

# Gluonts datasets

for dataset in ['electricity_nips', 'exchange_rate_nips', 'solar_nips', 'taxi_30min', 'traffic_nips', 'wiki-rolling_nips']:
  _ = get_dataset(dataset, regenerate=True)
