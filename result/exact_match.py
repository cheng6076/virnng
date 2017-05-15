import sys

f = sys.argv[1]
file = open(f, 'r')
f_lines = file.readlines()

ref = open('gold.txt', 'r')
ref_lines = ref.readlines()

count = 0
for lid, line in enumerate(f_lines):
   line = line.rstrip()
   ref_line = ref_lines[lid].rstrip()
   if line == ref_line: 
       count+=1 
   else:
       print ref_line
       print line

print count, len(f_lines), count*1.0/len(f_lines)

file.close()
ref.close()

