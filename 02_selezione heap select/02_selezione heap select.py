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

def input_array():
    return [int(x) for x in input().split(" ") if x]

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

a = input_array()
h = int(input()) - 1
#print(heapselect(a, h))
print(heapselect2(a, h))
