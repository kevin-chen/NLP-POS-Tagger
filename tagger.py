from os import stat
import numpy
import sys

def updateLikelihoodTable(likelihood, word, pos):
    posObject = likelihood.get(pos, None)
    if not posObject:
        posObject = dict()
        likelihood[pos] = posObject
    wordCount = posObject.get(word, 0) + 1
    posObject[word] = wordCount

def updateTransitionTable(transition, prevPos, currPos):
    prevPosObject = transition.get(prevPos, None)
    if not prevPosObject:
        prevPosObject = dict()
        transition[prevPos] = prevPosObject
    currPosCount = prevPosObject.get(currPos, 0) + 1
    prevPosObject[currPos] = currPosCount

def updateLikelihoodProbabilities(likelihood, partOfSpeech):
    for pos in likelihood:
        numPOSOccuringOnce = 0
        numPOS = partOfSpeech[pos]
        frequency = likelihood[pos]
        for freq in frequency:
            if frequency[freq] == 1: numPOSOccuringOnce += 1
            frequency[freq] = float(frequency[freq]) / float(numPOS)
        # print(pos, float(numPOSOccuringOnce), float(numPOS))
        if float(numPOSOccuringOnce) / float(numPOS) == 0:
            likelihood[pos]["UNKNOWN_WORD"] = 0.001
        else:
            likelihood[pos]["UNKNOWN_WORD"] = float(numPOSOccuringOnce) / float(numPOS)
        # print("Unknown Likelihood", likelihood[pos]["UNKNOWN_WORD"])

def updateTransitionProbabilities(transition, transitionCount):
    for state in transition:
    # if state != "End_Sent":
        numPrevPosOccuringOnce = 0
        # print(state)
        numPrevPos = transitionCount[state]
        frequency = transition[state]
        for freq in frequency:
            if frequency[freq] == 1: numPrevPosOccuringOnce += 1
            frequency[freq] = float(frequency[freq]) / float(numPrevPos)
        if float(numPrevPosOccuringOnce) / float(numPrevPos) == 0:
            transition[state]["UNKNOWN_WORD"] = 0.001
        else:
            transition[state]["UNKNOWN_WORD"] = float(numPrevPosOccuringOnce) / float(numPrevPos)
        # print("Unknown Transition", transition[state]["UNKNOWN_WORD"])

def viterbi(tokens, likelihood, transition):
    N = len(likelihood) # number of POS
    T = len(tokens) # number of token/words
    pos = list(likelihood.keys())
    maxArr = [[0 for j in range(T)] for i in range(N)] # used for DP algorithm using previous word's probability
    maxPosArr = [[0 for j in range(T)] for i in range(N)] # used to backtrack

    # runs viterbi on entire table 
    for word_index in range(T): # loop through each column
        word = tokens[word_index]
        for pos_index in range(N): # loop through each part of speech/row
            p = pos[pos_index]
            if word_index == 0: # first word in sentence has previous max of 1
                t = 1 / 1000 #  transition['Begin_Sent']["UNKNOWN_WORD"]
                if p in transition['Begin_Sent']:
                    t = transition['Begin_Sent'][p]
                l = 1 / 1000 # likelihood[p]["UNKNOWN_WORD"]
                if word in likelihood[p]:
                    l = likelihood[p][word]
                a = t * l
                # print(a, t, l)
                maxArr[pos_index][word_index] = a
            else:
                prev_column = word_index - 1
                max_list = []
                for prev_pos_index in range(N): # for each pos, loop through previous word's part of speech
                    prev_pos = pos[prev_pos_index]
                    prev_probability = maxArr[prev_pos_index][prev_column]
                    
                    if prev_probability != 0:
                        t = 1 / 1000 #  transition[prev_pos]["UNKNOWN_WORD"]
                        if p in transition[prev_pos]:
                            t = transition[prev_pos][p]
                        l = 1 / 1000 # likelihood[p]["UNKNOWN_WORD"]
                        if word in likelihood[p]:
                            l = likelihood[p][word]
                        # print(a, t, l)
                        a = prev_probability * t * l
                        max_list.append(a)
                    else:
                        max_list.append(0)
                max_index = max(range(len(max_list)), key=max_list.__getitem__) # need max index to backtrack to previous column that gave max probability
                maxArr[pos_index][word_index] = max(max_list)
                maxPosArr[pos_index][word_index] = ((max_index, prev_column), pos[max_index])
    
    # handle the end state
    max_list = []
    prev_column = T - 1
    for prev_pos_index in range(N):
        prev_pos = pos[prev_pos_index]
        prev_probability = maxArr[prev_pos_index][prev_column]
        if prev_probability != 0:
            t = 1 / 1000 # transition[prev_pos]["UNKNOWN_WORD"]
            if 'End_Sent' in transition[prev_pos]:
                t = transition[prev_pos]['End_Sent']
            a = prev_probability * t
            max_list.append(a)
        else:
            max_list.append(0)
    max_index = max(range(len(max_list)), key=max_list.__getitem__)

    # Backtrack, traverse to find best path of best probabilities
    row, col = max_index, prev_column
    outputPOSSequence = []
    outputPOSSequence.insert(0, pos[max_index])
    while col > 0:
        maxPos = maxPosArr[row][col]
        row = maxPos[0][0]
        col = maxPos[0][1]
        outputPOSSequence.insert(0, maxPos[1])

    # output the sequence of POS that has best probability for this sentence
    return outputPOSSequence

# given a word/line, update the likelihood of word being given POS and transition given previous POS
def updatePriors(likelihood, transition, line, prevPos, transitionCount, partOfSpeech):
    line = line.strip()
    if line == "":
        updateTransitionTable(transition, prevPos, 'End_Sent')
        return "Begin_Sent"
    else:
        parts = line.split("\t") # get the word and pos of line
        word = parts[0]
        pos = parts[1]
        partOfSpeech[pos] = partOfSpeech.get(pos, 0) + 1
        transitionCount[prevPos] = transitionCount.get(prevPos, 0) + 1
        updateLikelihoodTable(likelihood, word, pos) # update the likelihood hashtable
        updateTransitionTable(transition, prevPos, pos) # update the transition hashtable
        return pos

# reads the entire training file and updates each table
def trainingData(trainingFile, transition, likelihood, transitionCount, partOfSpeech):
    prevPos = "Begin_Sent"
    transition['Begin_Sent'] = dict()
    transition['End_Sent'] = dict()
    line = trainingFile.readline()
    while line != None:
        if len(line) == 0:
            break
        prevPos = updatePriors(likelihood, transition, line, prevPos, transitionCount, partOfSpeech)
        line = trainingFile.readline()
    # convert counts to probabilities
    updateLikelihoodProbabilities(likelihood, partOfSpeech)
    updateTransitionProbabilities(transition, transitionCount)

# read the development file and array output sentence of array words 
def createSentences(developmentFile):
    line = developmentFile.readline()
    output = []
    curr = []
    while line != None: # line == "" denotes new sentence
        if len(line) == 0:
            break
        line = line.strip()
        if line == "":
            output.append(curr)
            curr = []
        else:
            curr.append(line)
        line = developmentFile.readline()
    return output

# for each sentence, run viberti on that sentence and write file with each word and predicted pos tag
def tagger(allSentences, transition, likelihood):
    outputFile = open("RESULTS.pos", "w")
    for sentence in allSentences:
        pos_tagged = viterbi(sentence, likelihood, transition)
        for word_index in range(len(sentence)):
            try:
                pos = pos_tagged[word_index]
                word = sentence[word_index]
                content = word + "\t" + pos + "\n"
                outputFile.write(content)
            except:
                print(sentence, word_index)
        outputFile.write("\n")
    outputFile.close()

def main(args):
    trainingName = args[1]
    developmentName = args[2]

    likelihood = dict()
    transition = dict()
    transitionCount = dict()
    partOfSpeech = dict()

    trainingFile = open(trainingName, "r")
    trainingData(trainingFile, transition, likelihood, transitionCount, partOfSpeech)
    trainingFile.close()

    developmentFile = open(developmentName, "r")
    allSentences = createSentences(developmentFile)
    developmentFile.close()

    tagger(allSentences, transition, likelihood)


if __name__ == '__main__': sys.exit(main(sys.argv))