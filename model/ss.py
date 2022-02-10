import os
from subprocess import *

#run secondary script 1
p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\knnReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()
print(output[0])

#run secondary script 2
p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\mlpReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\lrReg.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\svmclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\knnclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()

p = Popen([r'Python', 'C:\\Users\\David\\OneDrive\\Desktop\\UF\\Academics\\Spring 2022\\CAP 4613\\project\\p1\\model\\mlpclassif.py'], shell=True, stdin=PIPE, stdout=PIPE)
output = p.communicate()