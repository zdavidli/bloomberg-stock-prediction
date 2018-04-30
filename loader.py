import csv
import numpy as np


print("Loading Data...")
datafile = open('nasdaq100/full/full_non_padding.csv', 'r');
tickers = None
data = []
prevrow = [0 for i in range(1000)]
for i, row in enumerate(csv.reader(datafile, delimiter=',')):
  if i == 0:
    tickers = row
  else:
    for i in range(len(row)):
      try:
        row[i] = float(row[i])
        prevrow[i] = row[i]
      except:
        row[i] = prevrow[i]
    data.append(row)

data = np.array(data).T
tickers = np.array(tickers)
print("Data Loaded!")