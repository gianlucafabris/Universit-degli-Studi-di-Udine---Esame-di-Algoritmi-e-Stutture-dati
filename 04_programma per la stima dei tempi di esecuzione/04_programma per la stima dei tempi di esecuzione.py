import time
import math
import random
import matplotlib.pyplot as plt

#selezione quick select
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

#selezione heap select
class MinHeap:
    heap = []

    def length(self):
        return len(self.heap)

    def left(self, i):
        j = 2*i+1
        if j >= self.length():
            return None
        return j

    def right(self, i):
        j = 2*i+2
        if j >= self.length():
            return None
        return j

    def getmin(self):
        assert len(self.heap) > 0, "getmin on empty heap"
        return self.heap[0]

    def extract(self):
        self.heap[0] = self.heap.pop()
        #analoga a:
        #self.heap[0] = self.heap[-1]
        #self.heap = self.heap[0:len(self.heap)-1]
        self.heapify(0)

    def insert(self, x):
        self.heap.append(x)
        self.moveup(len(self.heap)-1)

    def buildheap(self, a):
        self.heap = a.copy()
        for i in range(len(self.heap)-1, -1, -1):
            self.heapify(i)

    def change(self, i, x):
        assert i < len(self.heap), "changed called with wrong index"
        if x<self.heap[i]:
            self.heap[i] = x
            self.moveup(i)
        elif x>self.heap[i]:
            self.heap[i] = x
            self.heapify(i)

    def heapify(self, i):
        l = self.left(i)
        r = self.right(i)
        #dovrei prendere i minimo fra heap[i], heap[l] e heap[r] e lo scabio con heap[i]
        argmin = i
        if l and self.heap[l] < self.heap[argmin]:
            argmin = l
        if r and self.heap[r] < self.heap[argmin]:
            argmin = r
        if i != argmin:
            self.heap[i], self.heap[argmin] = self.heap[argmin], self.heap[i] #scambio
            self.heapify(argmin)

    def moveup(self, i):
        if i == 0:
            return
        p = (i+1)//2 - 1
        if self.heap[i] < self.heap[p]:
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i] #scabio
            self.moveup(p)

class MinHeapAux:
    heap = []

    def length(self):
        return len(self.heap)

    def left(self, i):
        j = 2*i+1
        if j >= self.length():
            return None
        return j

    def right(self, i):
        j = 2*i+2
        if j >= self.length():
            return None
        return j

    def getmin(self):
        assert len(self.heap) > 0, "getmin on empty heap"
        return self.heap[0]

    def extract(self):
        if(self.length() > 1):
            self.heap[0] = self.heap.pop()
            #analoga a:
            #self.heap[0] = self.heap[-1]
            #self.heap = self.heap[0:len(self.heap)-1]
            self.heapify(0)
        else:
            self.heap.pop()

    def insert(self, x):
        self.heap.append(x)
        self.moveup(len(self.heap)-1)

    def buildheap(self, a):
        self.heap = a.copy()
        for i in range(len(self.heap)-1, -1, -1):
            self.heapify(i)

    def change(self, i, x):
        assert i < len(self.heap), "changed called with wrong index"
        (x_val, x_aux) = x
        (i_val, i_aux) = self.heap[i]
        if x_val<i_val:
            self.heap[i] = x
            self.moveup(i)
        elif x_val>i_val:
            self.heap[i] = x
            self.heapify(i)

    def heapify(self, i):
        l = self.left(i)
        r = self.right(i)
        #dovrei prendere i minimo fra heap[i], heap[l] e heap[r] e lo scabio con heap[i]
        argmin = i
        (l_val, l_aux) = self.heap[l] if l else (None, None)
        (argmin_val, argmin_aux) = self.heap[argmin]
        if l and l_val < argmin_val:
            argmin = l
        (r_val, r_aux) = self.heap[r] if r else (None, None)
        (argmin_val, argmin_aux) = self.heap[argmin]
        if r and r_val < argmin_val:
            argmin = r
        if i != argmin:
            self.heap[i], self.heap[argmin] = self.heap[argmin], self.heap[i] #scambio
            self.heapify(argmin)

    def moveup(self, i):
        if i == 0:
            return
        p = (i+1)//2 - 1
        (i_val, i_aux) = self.heap[i]
        (p_val, p_aux) = self.heap[p]
        if self.heap[i] < self.heap[p]:
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i] #scabio
            self.moveup(p)

def heapselect(a, h): #complessità O(n + k log n)
    heap = MinHeap()
    heap.buildheap(a) #tempo O(n)
    for i in range(0, h): #h volte
        r = heap.getmin()
        heap.extract() #estraggo il minimo
    return heap.getmin()

#heap principale
# non la tocco
#heap ausilaria
# all'inizio copio la radice
# per k volte
#  -estraggo min heap ausiliaria
#  -aggiungo i figli della heap principale del elemento appena tolto
def heapselect2(a, h): #complessità O(n + k log k)
    main_heap = MinHeap()
    main_heap.buildheap(a) #tempo O(n)
    aux_heap = MinHeapAux()
    aux_heap.insert((main_heap.heap[0], 0))
    for i in range(0, h): #h volte
        (x, j) = aux_heap.getmin()
        aux_heap.extract()
        l = main_heap.left(j)
        r = main_heap.right(j)
        if l != None:
            aux_heap.insert((main_heap.heap[l], l))
        if r != None:
            aux_heap.insert((main_heap.heap[r], r))
    (x, j) = aux_heap.getmin()
    return x

#selezione median of medians select
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

#tempi
def resolution():
    start = time.monotonic()
    while time.monotonic() == start:
        pass
    stop = time.monotonic()
    return stop - start

def generate_input(n, maxv):
    a = [0]*n
    for i in range(n):
        a[i] = random.randint(0, maxv)
    return a

def benchmark(n, k, maxv, func, resolution, max_rel_error = 0.001):
    tmin = resolution*(1+(1/max_rel_error))
    count = 0
    start = time.monotonic()
    while time.monotonic() - start < tmin:
        a = generate_input(n, maxv)
        func(a, k-1)
        count += 1
    duration = time.monotonic() - start
    return duration/count

if __name__ == "__main__":
    resolution = resolution()
    nmin = 100
    nmax = 100000
    iters = 100
    base = math.e**((math.log(nmax)-math.log(nmin))/(iters-1))

    points = [(None, None, None, None, None)]*iters

    for i in range(iters):
        print(f"\t{i}", end='')
        n = int(base**i * nmin)
        k = random.randint(1, n)
        points[i] = (n,\
                    benchmark(n, k, 100, quickselect, resolution),\
                    benchmark(n, k, n, quickselect, resolution),\
                    benchmark(n, k, 2**500, quickselect, resolution),\
                    benchmark(n, k, 100, quickselect3way, resolution),\
                    benchmark(n, k, n, quickselect3way, resolution),\
                    benchmark(n, k, 2**500, quickselect3way, resolution),\
                    benchmark(n, k, 100, heapselect, resolution),\
                    benchmark(n, k, n, heapselect, resolution),\
                    benchmark(n, k, 2**500, heapselect, resolution),\
                    benchmark(n, k, 100, heapselect2, resolution),\
                    benchmark(n, k, n, heapselect2, resolution),\
                    benchmark(n, k, 2**500, heapselect2, resolution),\
                    benchmark(n, k, 100, mediansselect, resolution),\
                    benchmark(n, k, n, mediansselect, resolution),\
                    benchmark(n, k, 2**500, mediansselect, resolution))

    print("")

    xs, ysq1, ysq2, ysq3, ysq3w1, ysq3w2, ysq3w3, ysh1, ysh2, ysh3, ysh21, ysh22, ysh23, ysm1, ysm2, ysm3 = zip(*points)

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare.png')

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - quick.png')

    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - heap.png')

    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - medians.png')

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - max input 100.png')

    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - max input n.png')

    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.title("lineare")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('lineare - max input 2e500.png')

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog.png')

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - quick.png')

    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - heap.png')

    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - medians.png')

    plt.plot(xs, ysq3w1, label=f"quickselect3way 100")
    plt.plot(xs, ysh21, label=f"heapselect2 100")
    plt.plot(xs, ysm1, label=f"mediansselect 100")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - max input 100.png')

    plt.plot(xs, ysq3w2, label=f"quickselect3way n")
    plt.plot(xs, ysh22, label=f"heapselect2 n")
    plt.plot(xs, ysm2, label=f"mediansselect n")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - max input n.png')

    plt.plot(xs, ysq3w3, label=f"quickselect3way 2**500")
    plt.plot(xs, ysh23, label=f"heapselect2 2**500")
    plt.plot(xs, ysm3, label=f"mediansselect 2**500")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("loglog")
    plt.xlabel("n - dimensione input")
    plt.ylabel("t - tempo")
    plt.legend()
    plt.show()
    plt.savefig('loglog - max input 2e500.png')
