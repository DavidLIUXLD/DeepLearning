import os
from subprocess import *

#run secondary script 1
p = Popen([r'Python', 'model\\knnReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()
print(output[0])

#run secondary script 2
p = Popen([r'Python', 'model\\mlpReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'model\\lrReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'model\\svmclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'model\\knnclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'model\\mlpclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()
