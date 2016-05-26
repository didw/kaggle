import sys
import numpy as np

srcFile = 'data/train.csv'
head = 1
cnt = 0
userMaxId = 1198785
destMaxId = 65107
destList = np.zeros(destMaxId+1)
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
    destList[destId] = 1

    if 1:#words[20] == 1:
      cl_user_b[userId][cluster] += 1
      cl_dest_b[destId][cluster] += 1
    else:
      cl_user_c[userId][cluster] += 1
      cl_dest_c[destId][cluster] += 1

print('\n\n')



print('Evaluation Process...')
for w in [0]:
  cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1) + w*cl_user_c / cl_user_c.sum(axis=1).reshape(userMaxId+1, 1)
  cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1) + w*cl_dest_c / cl_dest_c.sum(axis=1).reshape(destMaxId+1, 1)

  score1 = 0
  score2 = 0
  score3 = 0
  score4 = 0
  score5 = 0
  head = 1
  cnt = 0
  N = 1
  with open(srcFile) as sf:
    print('\nClick Weight = {}'.format(w))
    for line in sf.readlines():
      cnt += 1
      #sys.stdout.write("\rProcessing.. %.2f s1:%.2f, s2:%.2f, s3:%.2f%%" % (cnt * 100.0 / 37670294, score1*100.0/N, score2*100.0/N, score3*100.0/N))
      sys.stdout.write("\rProcessing.. {0:2.2f} s1:{1:3.2f}, s2:{2:3.2f}, s3:{3:3.2f}, s4:{4:3.2f}, s5:{5:3.2f}".format(
        cnt * 100.0 / 37670294, score1*100.0/N, score2*100.0/N, score3*100.0/N, score4*100.0/N, score5*100.0/N))
      #sys.stdout.write("\rProcessing.. %.2f s1:%.2f, s2:%.2f, s3:%.2f%%" % (cnt * 100.0 / 100000, score1*100/N, score2*100/N, score3*100/N))
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

      cluster = cl_user[userId]
      pref = np.argsort(-cluster)
      for i in range(5):
        if pref[i] == clId:
          score1 += 1/(i+1)

      cluster = cl_dest[destId]
      pref = np.argsort(-cluster)
      for i in range(5):
        if pref[i] == clId:
          score2 += 1/(i+1)

      cluster = cl_user[userId] * cl_dest[destId]
      pref = np.argsort(-cluster)
      for i in range(5):
        if pref[i] == clId:
          score3 += 1/(i+1)

      cluster = np.array(map(lambda x:np.log(x), cl_user[userId])) + 2*np.array(map(lambda x:np.log(x), cl_dest[destId]))
      pref = np.argsort(-cluster)
      for i in range(5):
        if pref[i] == clId:
          score4 += 1/(i+1)

      cluster = 2*np.array(map(lambda x:np.log(x), cl_user[userId])) + np.array(map(lambda x:np.log(x), cl_dest[destId]))
      pref = np.argsort(-cluster)
      for i in range(5):
        if pref[i] == clId:
          score5 += 1/(i+1)

  print('\n\n')
  print('click weight: {0}, score(1:0): {1:2.2f}, score(0:1): {2:2.2f}, score(1:1): {3:2.2f}, score(weight(2:1)): {4:2.2f}, score(weight(1:2)): {5:2.2f}'.format(
        w, score1* 100.0 / N, score2* 100.0 / N, score3* 100.0 / N, score4* 100.0 / N, score5* 100.0 / N))

print('Done')



print('Make Submission file...')
testFile = 'data/test.csv'
subFile = 'submission/submission(1:0)_10.csv'
sf1 = open(subFile, 'w')
sf1.write("id,hotel_cluster\n")
subFile = 'submission/submission(0:1)_10.csv'
sf2 = open(subFile, 'w')
sf2.write("id,hotel_cluster\n")
subFile = 'submission/submission(1:1)_10.csv'
sf3 = open(subFile, 'w')
sf3.write("id,hotel_cluster\n")
subFile = 'submission/submission(2:1)_10.csv'
sf4 = open(subFile, 'w')
sf4.write("id,hotel_cluster\n")
subFile = 'submission/submission(1:2)_10.csv'
sf5 = open(subFile, 'w')
sf5.write("id,hotel_cluster\n")
for w in [0]:
  cl_user = cl_user_b / cl_user_b.sum(axis=1).reshape(userMaxId+1, 1)# + w*cl_user_c / cl_user_c.sum(axis=1).reshape(userMaxId+1, 1)
  cl_dest = cl_dest_b / cl_dest_b.sum(axis=1).reshape(destMaxId+1, 1)# + w*cl_dest_c / cl_dest_c.sum(axis=1).reshape(destMaxId+1, 1)

  score = 0
  head = 1
  cnt = 0
  N = 0
  with open(testFile) as tf:
    print('\nClick Weight = {}'.format(w))
    for line in tf.readlines():
      cnt += 1
      sys.stdout.write("\rProcessing.. %.2f%%" % (cnt * 100.0 / 2528240))
      sys.stdout.flush()
      if head == 1:
        head = 0
        continue

      words = line.split(',')

      Id = int(words[0])
      userId = int(words[8])
      destId = int(words[17])

      if destId > destMaxId or destId == '' or destList[destId] == 0:
        cluster = cl_user[userId]
      else:
        cluster = cl_user[userId]
      pref = np.argsort(-cluster)
      sf1.write("{},{} {} {} {} {}\n".format(Id, pref[0], pref[1], pref[2], pref[3], pref[4]))

      if destId > destMaxId or destId == '' or destList[destId] == 0:
        cluster = cl_user[userId]
      else:
        cluster = cl_dest[destId]
      pref = np.argsort(-cluster)
      sf2.write("{},{} {} {} {} {}\n".format(Id, pref[0], pref[1], pref[2], pref[3], pref[4]))

      if destId > destMaxId or destId == '' or destList[destId] == 0:
        cluster = cl_user[userId]
      else:
        cluster = cl_user[userId] * cl_dest[destId]
      pref = np.argsort(-cluster)
      sf3.write("{},{} {} {} {} {}\n".format(Id, pref[0], pref[1], pref[2], pref[3], pref[4]))

      if destId > destMaxId or destId == '' or destList[destId] == 0:
        cluster = cl_user[userId]
      else:
        cluster = np.array(map(lambda x:np.log(x), cl_user[userId])) + 2*np.array(map(lambda x:np.log(x), cl_dest[destId]))
      pref = np.argsort(-cluster)
      sf4.write("{},{} {} {} {} {}\n".format(Id, pref[0], pref[1], pref[2], pref[3], pref[4]))

      if destId > destMaxId or destId == '' or destList[destId] == 0:
        cluster = cl_user[userId]
      else:
        cluster = 2*np.array(map(lambda x:np.log(x), cl_user[userId])) + np.array(map(lambda x:np.log(x), cl_dest[destId]))
      pref = np.argsort(-cluster)
      sf5.write("{},{} {} {} {} {}\n".format(Id, pref[0], pref[1], pref[2], pref[3], pref[4]))

sf1.close()
sf2.close()
sf3.close()
sf4.close()
sf5.close()
print()
print('Done')

