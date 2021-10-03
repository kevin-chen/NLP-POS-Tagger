m = dict()
m["pos"] = dict()
m["pos"]["word"] = 2

print(m)
for pos in m:
    curr_pos = m[pos]
    for word in curr_pos:
        curr_pos[word] = float(curr_pos[word]) / float(3)
print(m)