import numpy

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

def updateLikelihoodProbabilities(likelihood):
    for pos in likelihood:
        count = 0
        frequency = likelihood[pos]
        for freq in frequency:
            count += frequency[freq]
        for freq in frequency:
            frequency[freq] = float(frequency[freq]) / float(count)

def updateTransitionProbabilities(transition):
    for state in transition:
        count = 0
        frequency = transition[state]
        for freq in frequency:
            count += frequency[freq]
        for freq in frequency:
            frequency[freq] = float(frequency[freq]) / float(count)

def viterbi(tokens, likelihood, transition):
    # N = number of POS
    # T = number of token/words
    N = len(likelihood)
    T = len(tokens)
    # create the 2d array, (N + 2) x T; POS by Words
    pos = list(likelihood.keys())
    maxArr = [[0 for j in range(T)] for i in range(N)]
    maxPosArr = [[0 for j in range(T)] for i in range(N)]

    # print(pos)

    for word_index in range(T): # loop through each column
        word = tokens[word_index]
        # print(word)
        for pos_index in range(N): # loop through each part of speech
            p = pos[pos_index]
            if word_index == 0:
                t = 0
                if p in transition['Begin_Sent']:
                    t = transition['Begin_Sent'][p]
                l = 0
                if word in likelihood[p]:
                    l = likelihood[p][word]
                a = t * l
                # print(p, t, l, a)
                maxArr[pos_index][word_index] = a
            else:
                prev_column = word_index - 1
                max_list = []
                for prev_pos_index in range(N): # for each pos, loop through previous word's part of speech
                    prev_pos = pos[prev_pos_index]
                    # print(word, prev_pos)
                    prev_probability = maxArr[prev_pos_index][prev_column]
                    # if word == "an":
                    #     print("Finding prev prob of", prev_pos, "to", p, "which is", prev_probability)
                    
                    if prev_probability != 0:
                        # print(prev_pos)
                        t = 0
                        if p in transition[prev_pos]:
                            t = transition[prev_pos][p]
                        l = 0
                        if word in likelihood[p]:
                            l = likelihood[p][word]
                        a = prev_probability * t * l
                        # if word == "an" and a != 0:
                        #     print("a:", a, t, l)
                        # if word == "an" and prev_pos == "IN" and p == "DT":
                        #     print(a, prev_probability, t, l, prev_pos, p)
                        # if word == "Haag":
                        #     print(prev_pos_index, pos_index, p, t, l)
                        max_list.append(a)
                    else:
                        max_list.append(0)
                max_index = max(range(len(max_list)), key=max_list.__getitem__)
                # print(pos[max_index])
                maxArr[pos_index][word_index] = max(max_list)
                maxPosArr[pos_index][word_index] = ((max_index, prev_column), pos[max_index])
                # print(p,len(max_list))
                # print(max(max_list))
    max_list = []
    prev_column = T - 1
    for prev_pos_index in range(N):
        prev_pos = pos[prev_pos_index]
        prev_probability = maxArr[prev_pos_index][prev_column]
        if prev_probability != 0:
            t = 0
            if 'End_Sent' in transition[prev_pos]:
                t = transition[prev_pos]['End_Sent']
            a = prev_probability * t
            # print(a, prev_probability, t)
            max_list.append(a)
        else:
            max_list.append(0)
    max_index = max(range(len(max_list)), key=max_list.__getitem__)
    # print(max_list, max_index)
    # print(pos[max_index])
    # maxArr[pos_index][word_index] = max(max_list)
    # maxPosArr[pos_index][word_index] = ((max_index, prev_column), pos[max_index])
    # print(max_index, pos[max_index])
    # print(len(max_list))
    # arr[pos_index][word_index] = max(max_list)
    
    # print(maxArr)

    # Backtrack
    row, col = max_index, prev_column
    outputPOSSequence = []
    # print(pos[max_index])
    outputPOSSequence.insert(0, pos[max_index])
    while col > 0:
        maxPos = maxPosArr[row][col]
        row = maxPos[0][0]
        col = maxPos[0][1]
        # print(maxPos[1])
        outputPOSSequence.insert(0, maxPos[1])

    # return maxPosArr
    return outputPOSSequence

def updatePriors(likelihood, transition, line, prevPos):
    line = line.strip()

    if line == "":
        updateTransitionTable(transition, prevPos, 'End_Sent')
        return "Begin_Sent"
    else:
        # get the word and pos of line
        parts = line.split("\t")
        word = parts[0]
        pos = parts[1]

        # update the likelihood hashtable
        updateLikelihoodTable(likelihood, word, pos)
        # update the transition hashtable
        updateTransitionTable(transition, prevPos, pos)
        
        return pos

def trainingData(trainingFile, transition, likelihood):
    prevPos = "Begin_Sent"
    transition['Begin_Sent'] = dict()
    transition['End_Sent'] = dict()

    line = trainingFile.readline()

    while line != None:
        if len(line) == 0:
            break
        prevPos = updatePriors(likelihood, transition, line, prevPos)
        line = trainingFile.readline()
    
    updateLikelihoodProbabilities(likelihood)
    updateTransitionProbabilities(transition)

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

def main():
    likelihood = dict()
    transition = dict()

    trainingFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos", "r")
    # trainingFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/shorter.pos", "r")
    trainingData(trainingFile, transition, likelihood)
    trainingFile.close()

    # maxPosArr = viterbi(["Ms.", "Haag", "plays", "Elianti", "."], likelihood, transition)

    # print("Likelihood", likelihood)
    # print("")
    # print("Transition", transition)

    developmentFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words", "r")
    # developmentFile = open("WSJ_POS_CORPUS_FOR_STUDENTS/shorter.words", "r")
    allSentences = createSentences(developmentFile)
    developmentFile.close()

    tagger(allSentences, transition, likelihood)
    # print(viterbi(allSentences[0], likelihood, transition))


if __name__ == "__main__":
    main()