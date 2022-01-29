import os

prefix = 'img'
suffix = '.jpg'
input_dir = './res_processed_0'
files = os.listdir(input_dir)
count = 0
for file in files:
    os.rename(os.path.join(input_dir, file),
              os.path.join(input_dir, prefix + str(count) + suffix))
    count = count + 1
