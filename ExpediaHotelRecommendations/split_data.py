srcFile = 'data/train.csv'

desFile = ['' for i in range(100)]
df = ['' for i in range(100)]
for i in range(100):
  desFile[i] = 'data/train/train_{0:02d}.csv'.format(i)
  df[i] = open(desFile[i], 'w')



head = 1
with open(srcFile) as sf:
  for line in sf.readlines():
    if head == 1:
      head = 0
      continue
    words = line.split(',')
    idx = int(words[23])
    df[idx].write(line)

for i in range(100):
  df[i].close()