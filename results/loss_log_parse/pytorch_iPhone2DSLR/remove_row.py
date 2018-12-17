#!/usr/bin/env python3

import csv
import sys

with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile)
    next(reader)
    with open(sys.argv[1] + '.fixed', 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['Epoch', 'Step', 'Value'])
        writer.writeheader()
        for row in reader:
            writer.writerow(row)

