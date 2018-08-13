from timeit import default_timer as timer

filename = input("Enter filename: ")
start = timer()
file = open(filename, encoding='utf-8')
fileLines = []
i = 0
for line in file:
    if line.strip() != "":
        fileLines.append(line.strip())
    i += 1
startFound = False
startLine = 0
for i in range(len(fileLines)):
    if fileLines[i].startswith("***"):
        if not startFound:
            startLine = i
            startFound = True
        else:
            endLine = i-1
            break
fileLines = fileLines[startLine+1:endLine]
file.close()
file = open(filename, "w")
for line in fileLines:
    file.write(line + "\n")
file.close()
end = timer()
print("File processed in " + str((end-start)/1000) + " seconds")
