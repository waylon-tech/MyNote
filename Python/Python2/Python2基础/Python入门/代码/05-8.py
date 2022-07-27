tg = [1,2,3,4,5,6,7,8,9]
ts = [0,1,2,3,4,5,6,7,8]
for tg1 in tg:
    for ts1 in tg:
        if ts1 > tg1:
            print tg1 * 10 + ts1
