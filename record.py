import csv
import os

with open("./data/record.csv", "w") as f:
    writer = csv.writer(f)
    fs = os.listdir("./data/img")
    for item in fs:
        speed = item.split("_")[1]
        if float(speed) < 0.0:
            continue

        angle = item.split("_")[0]
        writer.writerow([angle, item])