import csv
import os

os.chdir('/home/qy2672/learning-alternations/data/datives/ditransitive')
input_file = './non-ditransitive.csv'
output_file = '../../training_sets/ditransitives_removed/train.txt'

with open(input_file, mode='r', newline='') as csvfile:
    csvreader = csv.DictReader(csvfile)
    with open(output_file, mode='w') as txtfile:
        for row in csvreader:
            txtfile.write(row['sentence'] + '\n')