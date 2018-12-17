#!/usr/bin/env python3

import csv
import sys

with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile)
    with open(sys.argv[1] + '.fixed', 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['Epoch', 'Value'])
        writer.writeheader()
        for row in reader:
            epoch = float(row['Epoch']) + (float(row['Step']) / 1231.0)
            writer.writerow({'Epoch': epoch, 'Value': row['Value']})
