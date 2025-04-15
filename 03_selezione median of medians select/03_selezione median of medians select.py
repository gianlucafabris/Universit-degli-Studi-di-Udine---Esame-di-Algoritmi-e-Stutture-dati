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

def quicksort(a, i = 0, j = None):
    if j == None:
        j = len(a)
    #caso base
    if j-i <= 1:
        return
    #caso induttivo
    k = partition(a, i, j)
    # 0     k     j
    #|≤|≤|≤|p|>|>|
    quicksort(a, i, k)
    quicksort(a, k, j)

def quicksort3way(a, i = 0, j = None):
    if j == None:
        j = len(a)
    #caso base
    if j-i <= 1:
        return
    #caso induttivo
    k, h = partition3way(a, i, j)
    # 0     k     j
    #|<|<|p|p|p|>|
    quicksort3way(a, i, k)
    quicksort3way(a, h, j)

def split5(a):
    b = []
    for i in range(0, len(a), 5):
        b.append(a[i:i+5])
    return b

def partitionWithEl(a, el):
    pivot_index = a.index(el)
    a[pivot_index], a[len(a)-1] = a[len(a)-1], a[pivot_index]
    return partition(a, 0, len(a))

def partition3wayWithEl(a, el):
    pivot_index = a.index(el)
    a[pivot_index], a[len(a)-1] = a[len(a)-1], a[pivot_index]
    return partition3way(a, 0, len(a))

def mediansselect(a, h):
    if len(a)<=5:
        quicksort3way(a)
        return a[h]
    #(questi passi possono essere fatti in place)
    temp = split5(a)
    #ordino sottoarray
    for i in range(len(temp)):
        quicksort3way(temp[i]) #costo costante con input 5 elementi
    #array dei mediani
    b = [x[(len(x)-1)//2] for x in temp]
    #mediano
    M = mediansselect(b, len(b)//2)
    #partizione e ricorsione
    x = partitionWithEl(a, M)
    # 0     x     len(a)
    #|≤|≤|≤|M|>|>|
    if h == x:
        return M
    elif h < x:
        return mediansselect(a[0:x], h)
    elif h > x:
        return mediansselect(a[x:len(a)], h-x)

a = input_array()
h = int(input()) - 1
print(mediansselect(a, h))
