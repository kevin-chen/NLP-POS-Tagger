How to run: python3 main_kc3585_HW3.py trainingFile.pos developmentFile.words

Handle OOV items: Used the first method of constant 1/1000 as likelihood. I implemented the third option of using a distribution of 
all items occuring once as the basis for computing likelihood for OOV items, but I did not have enough time to debug the issue where
all parts of speech were tagged incorrectly with "FW".
This implementation counted up the number of words that occurred once for a part of speech and counted up total number of words in 
the same part of speech, and divided to get UNKNOWN_WORD probability for a single word with that part of speech.

Viterbi Implementation: uses hashmap for likelihood and transition tables. Used a 2D array for dynamic programming style to figure out
the best probability and backtracking from the end state to start state.

Runtime: running the combined training and development file took less than 10 seconds.