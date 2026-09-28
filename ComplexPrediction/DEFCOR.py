
def cor(A, B):
    # A = [1, 2, 3, 4, 4]
    # B = [3, 4, 5, 6, 7]

    n = len(A)
    Aav = sum(A)/n
    Bav = sum(B)/n
    A1 = []
    Sum1 = 0
    B1 = []
    M = []
    Uqa = []
    Sum2 = 0
    Uqb = []
    Sum3 = 0

    for i in range(n):
        q1 = A[i] - Aav
        A1.append(q1)

        q2 = B[i] - Bav
        B1.append(q2)

        m = q1 * q2
        M.append(m)
        Sum1 = Sum1 + m

        Ua = q1 * q1
        Uqa.append(Ua)
        Sum2 = Sum2 + Ua

        Ub = q2 * q2
        Uqb.append(Ub)
        Sum3 = Sum3 + Ub

    if Sum2 * Sum3 != 0:
        Answer = Sum1/((Sum2 * Sum3)**0.5)

    else:
        Answer = 0



    # print(Answer)
    return(Answer)





