file = open("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos", "r")
first_line = file.readline().strip()
arr = first_line.split("\t")
print(arr)
file.close()