import sys

srcFile = 'data/train.csv'

desFile = ['' for i in range(100)]
df = ['' for i in range(100)]
for i in range(100):
  desFile[i] = 'data/train_click/train_{0:02d}.csv'.format(i)
  df[i] = open(desFile[i], 'w')



head = 1
cnt = 0
with open(srcFile) as sf:
  print('Load data from disk...')
  for line in sf.readlines():
    cnt += 1
    sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 37670294))
    sys.stdout.flush()
    if head == 1:
      head = 0
      continue
    words = line.split(',')
    if int(words[18]) == 1:
      continue
    dataMissing = 0
    for w in words:
      if w == '':
        dataMissing = 1
    if dataMissing == 1:
      continue
    idx = int(words[23])
    df[idx].write(line)

print()
print('File Closing..')

for i in range(100):
  df[i].close()

print('Done')
