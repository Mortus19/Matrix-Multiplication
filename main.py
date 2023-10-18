import os
import sys
import time
total_file = open("C:/Users/Zver/Source/Repos/Project4/x64/Release/my_mult_simdlen 1.txt",'w')
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


for N in range(200,2000,100):
    best_time = 1000000000000000.0
    path = f'C:/Users/Zver/Source/Repos/Project4/x64/Release/Generator.exe'
    path += ' '
    path += str(N)
    os.system(path)
    print(N)
    for i in range(6):
        os.system('cd C:\Program Files (x86)\Intel\oneAPI && setvars.bat && C:/Users/Zver/Source/Repos/Project4/x64/Release/mkl_mult.exe')
        os.system('C:/Users/Zver/Source/Repos/Project4/x64/Release/Matrix.exe')
        file = open("C:/Users/Zver/Source/Repos/Project4/x64/Release/checker_verdict.txt",'r')
        verdict = file.read()
        if verdict != "Correct!\n":
            print("NOT CORRECT!!!")
            sys.exit()
        file = open("C:/Users/Zver/Source/Repos/Project4/x64/Release/output.txt",'r')
        for line in file:
            cur_time = float(line)
            best_time = min(best_time,cur_time)
    total_file.write(f"{N} {toFixed(best_time,10)}\n")

for N in range(2000,5000 + 1,500):
    best_time = 1000000000000000.0
    path = f'C:/Users/Zver/Source/Repos/Project4/x64/Release/Generator.exe'
    path += ' '
    path += str(N)
    os.system(path)
    print(N)
    for i in range(6):
        os.system('cd C:\Program Files (x86)\Intel\oneAPI && setvars.bat && C:/Users/Zver/Source/Repos/Project4/x64/Release/mkl_mult.exe')
        os.system('C:/Users/Zver/Source/Repos/Project4/x64/Release/Matrix.exe')
        file = open("C:/Users/Zver/Source/Repos/Project4/x64/Release/checker_verdict.txt", 'r')
        verdict = file.read()
        if verdict != "Correct!\n":
            print("NOT CORRECT!!!")
            sys.exit()
        file = open("C:/Users/Zver/Source/Repos/Project4/x64/Release/output.txt", 'r')
        for line in file:
            cur_time = float(line)
            best_time = min(best_time, cur_time)
    total_file.write(f"{N} {toFixed(best_time, 10)}\n")