# file = open("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos", "r")
# first_line = file.readline().strip()
# arr = first_line.split("\t")
# print(arr)
# file.close()

likelihood = dict()
transition = dict()

# trainingFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos", "r")
trainingFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/shorter.pos", "r")
line = trainingFile.readline()

prevChar = ""
prevPos = "Begin_Sent"
transition['Begin_Sent'] = dict()
transition['End_Sent'] = dict()

def updateLikelihoodTable(word, pos):
    posObject = likelihood.get(pos, None)
    if not posObject:
        posObject = dict()
        likelihood[pos] = posObject
    wordCount = posObject.get(word, 0) + 1
    posObject[word] = wordCount

# take the previous POS and update the current POS by 1
def updateTransitionTable(prevPos, currPos):
    prevPosObject = transition.get(prevPos, None)
    if not prevPosObject:
        prevPosObject = dict()
        transition[prevPos] = prevPosObject
    currPosCount = prevPosObject.get(currPos, 0) + 1
    prevPosObject[currPos] = currPosCount

while line != None:
    if len(line) == 0:
        break
    
    line = line.strip()

    if line == "":
        updateTransitionTable(prevPos, 'End_Sent')
        prevPos = "Begin_Sent"
    else:
        # get the word and pos of line
        parts = line.split("\t")
        word = parts[0]
        pos = parts[1]

        # update the likelihood hashtable
        updateLikelihoodTable(word, pos)

        # update the transition hashtable
        updateTransitionTable(prevPos, pos)
        prevPos = pos

    # read the next line in the file
    line = trainingFile.readline()

print(likelihood)
print("\n", transition)
trainingFile.close()

# track previous pos, 