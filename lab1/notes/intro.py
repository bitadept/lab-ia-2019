def print_sep():
    print('-----------')

# v_1 = 5
# print(v_1)
# print(type(v_1))
#
# v_1 += 3
# print(v_1)

# lists
l_1 = [1, 5, 2, 15, 13]
print(l_1[3])
print(len(l_1))
print(l_1[-1])

l_1.append(10)
l_1.append([20, 30])
l_1.pop()
l_1.pop(1)

print(l_1)

l_1.extend([20, 30, 40])
print(l_1)

l_1.insert(1, 5000)
print(l_1)

l_1 = [1, 5, 2, 15, 13]
print(l_1[1:3])

print_sep()

print(l_1[1:-1:2])
print(l_1[1::2])
print(l_1[::2])

print_sep()

print(l_1)
a = b = c = 0
for val in l_1:
    if val < 7:
        a += val
    elif val > 7:
        b += val
    else:
        c += 1

print(f'{a}, {b}, {c}')

print_sep()

l_2 = []
for val in l_1:
    for val2 in l_1:
        if val2 > 7:
            l_2.append(val * val2)
print(l_2)

print_sep()

l_3 = [val * val2 for val in l_1 for val2 in l_1 if val2 > 7]
print(l_3)

print_sep()

for idx in range(len(l_1)):
# range(3) => 0, 1, 2
# range(1, 4) => 1, 2, 3
# range(0, 5, 2) => 0, 2, 4
    print(l_1[idx])

# functions
print(l_1)
def extrage(L):
    elem = L[-1]
    L.pop()
    return elem
print(extrage(l_1))
print(l_1)

print_sep()

L = l_1.copy()
print(extrage(l_1))
print(l_1)
print(L)

print_sep()
L = l_1
print(extrage(L))
print(L)
print(l_1)