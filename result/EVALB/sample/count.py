import re
f1 = open('sample.gld', 'r')
f2 = open('sample.tst', 'r')

s1 = []
s2 = []
for line in f1.readlines():
    s1.append(line.rstrip())
f1.close()
for line in f2.readlines():
    s2.append(line.rstrip())
f2.close()

count = 0
total = 0
wcount = 0
for i in range(len(s1)):

    line1 = s1[i]
    line2 = s2[i]
    line1 = re.sub(r'\(XX \w+\)', 'XX', line1)
    line2 = re.sub(r'\(XX \w+\)', 'XX', line2)
    line1 = line1.replace(')', ' )')
    line2 = line2.replace(')', ' )')
    tmp1 = []
    ns = 0
    line1 = line1.split(' ')
    for word in line1:
        tmp1.append(ns)
        if 'XX' in word : ns = ns+1
                   
    ns2 = 0
    line2 = line2.split(' ')
    for n, word in enumerate(line2):
        dn = min(n, len(tmp1)-1)
        total = total + 1
        if ns2 == tmp1[dn]:count = count+1
        if 'XX' in word: 
            ns2 = ns2+1
print (count*1.0/total)

        
