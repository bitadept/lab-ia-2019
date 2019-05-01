def println():
    print('##########################')

# 1 - with dictionary
def make_dict(L):
    d = {}
    for val in L:
        if val in d:
            d[val] = d[val] + 1
        else:
            d[val] = 1

    return d

print(make_dict([1,1,2,3,2,2,2,4,3,1,4]))

# 1 - with set
def make_dict2(L):
    l_unique = set(L)
    d = {}
    for val in l_unique:
        d[val] = L.count(val)
    return d

print(make_dict2([1,1,2,3,2,2,2,4,3,1,4]))

println()

print(set([1, 2, 3, 12, 5, 6, 3, 2]))

println()

# 2
def comune_necomune(L1, L2):
    l1_unique = list(set(L1))
    l2_unique = list(set(L2))
    l1_unique.extend(l2_unique)
    d = make_dict2(l1_unique)

    comune = []
    necomune = []

    for key in d:
        if d[key] == 1:
            necomune.append(key)
        else:
            comune.append(key)
    return comune, necomune

print(comune_necomune([1, 2, 3, 1, 3, 5, 2], [2, 3, 5, 2]))

println()

import random

# 3

# def rock_paper_scissors():
#     options = ['rock', 'paper', 'scissors']
#     computer_option = options[random.randint(0, 2)]
#     human_option = input('Choose: rock, paper, scissors\n')
#
#     print('Computer chose: ' + computer_option)
#
#     if computer_option == human_option:
#         print('It\'s a draw!')
#     elif computer_option == 'rock' and human_option == 'scissors':
#         print('You lost!')
#     elif computer_option == 'paper' and human_option == 'rock':
#         print('You lost!')
#     elif computer_option == 'scissors' and human_option == 'paper':
#         print('You lost!')
#     else:
#         print('You won!')
#
# rock_paper_scissors()
#
# println()
#
# # 3 - with tuples
# def rock_paper_scissors():
#     options = ['rock', 'paper', 'scissors']
#     computer_option = options[random.randint(0, 2)]
#     human_option = input('Choose: rock, paper, scissors\n')
#
#     print('Computer chose: ' + computer_option)
#
#     if computer_option == human_option:
#         print('It\'s a draw!')
#     elif (computer_option, human_option) in [('rock', 'scissors'), ('paper', 'rock'),
#                                              ('scissors', 'paper')]:
#         print('You lost!')
#     else:
#         print('You won!')
#
# rock_paper_scissors()

println()

# 4
def cows_and_bulls():
    # generez nr calculatorului
    nr_calc = random.randint(1000, 9999)
    nr_calc_verif = set(list(str(nr_calc)))

    while len(nr_calc_verif) < 4:
        nr_calc = random.randint(1000, 9999)
        nr_calc_verif = set(list(str(nr_calc)))

    nr_calc = list(str(nr_calc))

    # prelucrez nr dat de om
    while True:
        nr_om = list(str(input('Ghiceste numarul!')))
        bulls = cows = 0
        for ind in range(len(nr_om)):
            if nr_om[ind] == nr_calc[ind]:
                cows += 1
            else:
                bulls += nr_calc.count(nr_om[ind])

        if cows == 4:
            break
        else:
            print(f'cows: {cows}, bulls: {bulls}')

cows_and_bulls()