python3 sandbox.py

Testing development answer corpus with predicted pos: 
python3 WSJ_POS_CORPUS_FOR_STUDENTS/score.py WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.pos RESULTS.pos

Runs full training data and test on development set:
python3 tagger.py WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words

Runs smaller training data and test on smaller development set:
python3 tagger.py WSJ_POS_CORPUS_FOR_STUDENTS/shorter.pos WSJ_POS_CORPUS_FOR_STUDENTS/shorter.words

First Time: 
21273 out of 32853 tags correct
  accuracy: 64.752077

Second Time: (1 / 1000 likelihood)
22406 out of 32853 tags correct
  accuracy: 68.200773

