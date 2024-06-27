def input_array():
    return [int(x) for x in input().split(" ") if x]

def partition(a, i, j):
    p = a[j-1]
    k = i
    h = i
    while 0 <= k <= h < j:
        # 0       k   h   j-1
        #|≤|≤|≤|≤|>|>|?|?|p|
        if a[h] <= p:
            a[k], a[h] = a[h], a[k]
            k += 1
        h += 1
    # 0     k     h=j
    #|≤|≤|≤|p|>|>|
    return k-1

def partition3way(a, i, j):
    p = a[j-1]
    k = i
    h = i
    m = i
    while 0 <= k <= h <= m < j:
        # 0       k   h   m   j-1
        #|<|<|<|<|=|=|>|>|?|?|p|
        if a[m] < p:
            if k < h < m:
                a[k], a[h], a[m] = a[m], a[k], a[h]
            elif k == h < m:
                a[k], a[m] = a[m], a[k]
            elif k < h == m:
                a[k], a[h] = a[h], a[k]
            k += 1
            h += 1
        elif a[m] == p:
            a[h], a[m] = a[m], a[h]
            h += 1
        m += 1
    # 0   k     h j
    #|<|<|p|p|p|>|
    return k, h

def quickselect(a, h, i = 0, j = None):
    if j == None:
        j = len(a)
    #invariante: h dovrebbe sempre essere compreso tra i (incluso) e j (escluso)
    #caso base
    if j-i == 1 and h == i:
        return a[i]
    elif i == j:
        return None
    #caso induttivo
    k = partition(a, i, j)
    # 0     k     j
    #|≤|≤|≤|p|>|>|
    if i <= h < k:
        return quickselect(a, h, i, k)
    elif k <= h < j:
        return quickselect(a, h, k, j)

def quickselect3way(a, h, i = 0, j = None):
    if j == None:
        j = len(a)
    #caso base
    if j-i == 1 and h == i:
        return a[i]
    elif i == j:
        return None
    #caso induttivo
    k, l = partition3way(a, i, j)
    # 0     k     j
    #|<|<|p|p|p|>|
    if i <= h < k:
        return quickselect3way(a, h, i, k)
    elif k <= h < l:
        return a[k]
    elif l <= h < j:
        return quickselect3way(a, h, l, j)

a = input_array()
h = int(input()) - 1
#print(quickselect(a, h))
print(quickselect3way(a, h))
