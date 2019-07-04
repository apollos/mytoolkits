import os
import numpy

filename2score = lambda x: x[:x.rfind('.')].split('_')[-1]

img_files = sorted(os.listdir('samples'))

with open('train.txt', 'w') as train_txt:
    for f in img_files[:10000]:
        score = filename2score(f)
        line = 'samples/{} {}\n'.format(f, score)
        train_txt.write(line)

with open('val.txt', 'w') as val_txt:
    for f in img_files[10000:11000]:
        score = filename2score(f)
        line = 'samples/{} {}\n'.format(f, score)
        val_txt.write(line)

with open('test.txt', 'w') as test_txt:
    for f in img_files[11000:]:
        line = 'samples/{}\n'.format(f)
        test_txt.write(line)
