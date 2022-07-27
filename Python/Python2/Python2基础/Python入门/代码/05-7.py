sum = 0
x = 1
while True:
    if x % 2 == 0:
        x = x + 1
        continue
    if x > 100:
        break
    else:
        sum = sum + x
        x = x + 1
print sum
