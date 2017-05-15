import sys
f = sys.argv[1]
rf = open(f, 'r')
o = open(f+'eval', 'w')

for line in rf.readlines():
  line = line.strip().split(' ')
  p = ''
  for word in line:
    if not ('(' in word or ')' in word):
      p = p + '(POS '+word +')'+' '
    else:
      p = p + word+' '
  p = p.replace(' )', ')')
  o.write(p+'\n')

rf.close()
o.close()
