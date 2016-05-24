import sys
import numpy as np

srcFile = 'data/train.csv'


head = 1
cnt = 0
userMaxId = 1198785
destMaxId = 65107
# Recommend based on user id 1,198,785 x 100 = 119,878,500 (~500MB)
# Recommend based on dest id 65,107 x 100 = 6,510,700 (~25MB)
cl_user_b = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
cl_dest_b = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)
cl_user_c = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
cl_dest_c = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)
cl_user = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
cl_dest = np.ones((destMaxId+1)*101, dtype='f').reshape(destMaxId+1, 101)
with open(srcFile) as sf:
  print('Loading file...')
  for line in sf.readlines():
    cnt += 1
    sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 37670294))
    #sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 100000))
    sys.stdout.flush()
    if head == 1:
      head = 0
      continue

    words = line.split(',')    
    userId = int(words[7])
    destId = int(words[16])
    cluster = int(words[23])

    if words[20] == 1:
      cl_user_b[userId][cluster] += 1
      cl_dest_b[destId][cluster] += 1
    else:
      cl_user_c[userId][cluster] += 1
      cl_dest_c[destId][cluster] += 1

print('\n\n')



print('Evaluation Process...')
for w in [1, 0.1, 10, 0.3, 4]:
  cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1) + w*cl_user_c / cl_user_c.sum(axis=1).reshape(userMaxId+1, 1)
  cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1) + w*cl_dest_c / cl_dest_c.sum(axis=1).reshape(destMaxId+1, 1)

  score = 0
  head = 1
  cnt = 0
  N = 0
  with open(srcFile) as sf:
    print('\nClick Weight = {}'.format(w))
    for line in sf.readlines():
      cnt += 1
      sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 37670294))
      #sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 100000))
      sys.stdout.flush()
      if head == 1:
        head = 0
        continue

      words = line.split(',')

      if words[20] == 0:
        continue
      N += 1

      userId = int(words[7])
      destId = int(words[16])
      clId = int(words[23])

      cluster = cl_user[userId] + cl_dest[destId]
      pref = np.argsort(-cluster)

      for i in range(5):
        if pref[i] == clId:
          score += 1/(i+1)

  print('\n\n')
  print('click weight: {}, score: {}'.format(w, score * 100 / N))

print('Done')
