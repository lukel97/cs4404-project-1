Submitting jobs:

```bash
gcloud ml-engine jobs submit training cyclegan_summer2winter_yosemite_N \ 
--job-dir=gs://cs4404-a1-mlengine/cyclegan_summer2winter_yosemite \
--runtime-version 1.9 \
--module-name=version1.main \
--package-path=version1 \
--region=europe-west1 \
--scale-tier=BASIC_GPU -- \
--data-dir=gs://cs4404-a1-mlengine/datasets/summer2winter_yosemite/
```
