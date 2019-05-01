GOL = '#'


def __str__(matr):
    sir = (" ".join([str(x) for x in matr[0:3]]) + "\n" +
           " ".join([str(x) for x in matr[3:6]]) + "\n" +
           " ".join([str(x) for x in matr[6:9]]) + "\n")

    return sir


def elem_identice(lista):
    '''
    lista contine elementele de pe linie, coloana  sau
     diagonale si verifica daca are doar valori de x
    sau doar valori de 0

    Daca lista contine un castigator, il intoarce pe acesta (x sau 0), altfel intoarce False
    '''
    el_unice = list(set(lista))
    if len(el_unice) == 1 and el_unice[0] != GOL:
        return el_unice[0]
    return False


def final(matr):
    '''
    verifica liniile, coloanele si diagonalele cu ajutorul lui elem_identice si intoarce, dupa caz, castigatorul,
    remiza, sau False
    '''
    rezultate = [elem_identice(matr[0:3]), elem_identice(matr[3:6]), elem_identice(matr[6:9]), # linii
                 elem_identice([matr[0], matr[3], matr[6]]), elem_identice([matr[1], matr[4], matr[7]]), elem_identice([matr[2], matr[5], matr[8]]), # coloane
                 elem_identice([matr[0], matr[4], matr[8]]), elem_identice([matr[2], matr[4], matr[6]])] # diagonale
    if 'x' in rezultate:
        return 'x'
    elif '0' in rezultate:
        return '0'
    elif '#' not in matr:
        return 'remiza'

    return False

def afis_daca_final(stare_curenta):
    final_ans = final(stare_curenta)
    if (final_ans):
        if (final_ans == "remiza"):
            print("Remiza!")
        else:
            print("A castigat " + final_ans)

        return True

    return False

def schimba_j(juc):
    if juc == 'x':
        return '0'
    return 'x'

def main():
    print('Incepe jocul x si 0')

    # initializare jucatori
    raspuns_valid = False
    while not raspuns_valid:
        JMIN = input("Doriti sa jucati cu x sau cu 0? ").lower()
        if (JMIN in ['x', '0']):
            raspuns_valid = True
        else:
            print("Raspunsul trebuie sa fie x sau 0.")
    JMAX = '0' if JMIN == 'x' else 'x'

    # initializare tabla
    tabla_curenta = [GOL] * 9
    print("Tabla initiala")
    print(__str__(tabla_curenta))

    # creare stare initiala
    j_curent = 'x'

    while True:
        if j_curent == JMIN:
            # muta jucatorul 'x'

            while True:
                linie = int(input('linie: '))
                coloana = int(input('coloana: '))

                if 0 <= linie <= 2 and 0 <= coloana <= 2 and tabla_curenta[linie * 3 + coloana] == GOL:
                    break

                print('invalide')

            # In punctul acesta sigur am valide atat linia cat si coloana
            # deci pot plasa simbolul pe "tabla de joc"

            tabla_curenta[linie * 3 + coloana] = j_curent
            j_curent = schimba_j(j_curent)

            # afisarea starii jocului in urma mutarii utilizatorului
            print("\nTabla dupa mutarea jucatorului")
            print(__str__(tabla_curenta))

            # testez daca jocul a ajuns intr-o stare finala
            # si afisez un mesaj corespunzator in caz ca da
            if (afis_daca_final(tabla_curenta)):
                break

            # S-a realizat o mutare. Schimb jucatorul cu cel opus
            j_curent = '0'

        # --------------------------------
        else:  # jucatorul e JMAX
            if j_curent == JMAX:
                # muta jucatorul '0'

                while True:
                    linie = int(input('linie: '))
                    coloana = int(input('coloana: '))

                    if 0 <= linie <= 2 and 0 <= coloana <= 2 and tabla_curenta[linie * 3 + coloana] == GOL:
                        break

                    print('invalide')

                # In punctul acesta sigur am valide atat linia cat si coloana
                # deci pot plasa simbolul pe "tabla de joc"

                tabla_curenta[linie * 3 + coloana] = j_curent
                j_curent = schimba_j(j_curent)

                # afisarea starii jocului in urma mutarii utilizatorului
                print("\nTabla dupa mutarea jucatorului")
                print(__str__(tabla_curenta))

                # testez daca jocul a ajuns intr-o stare finala
                # si afisez un mesaj corespunzator in caz ca da
                if (afis_daca_final(tabla_curenta)):
                    break

                # S-a realizat o mutare. Schimb jucatorul cu cel opus
                j_curent = 'x'


if __name__ == "__main__":
    main()
