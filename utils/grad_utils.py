import heapq


class HeapItem:
    def __init__(self, item, reversed=False):
        self.item = item
        self.reversed = reversed

    def __lt__(self, other):
        return self.reversed ^ (self.item < other.item)


class Heap:
    def __init__(self, is_max_heap=False):
        self.is_max_heap = is_max_heap
        self.heap = []
        self._len = 0

    def empty(self):
        return len(self.heap) == 0

    def top(self):
        return self.heap[0].item

    def push(self, x):
        self._len += 1
        heapq.heappush(self.heap, HeapItem(x, self.is_max_heap))

    def pop(self):
        self._len -= 1
        return heapq.heappop(self.heap).item

    def __len__(self):
        return self._len


class IterativePercentile():
    def __init__(self, p):
        self.p = p / 100
        self.min_heap = Heap(is_max_heap=False)
        self.max_heap = Heap(is_max_heap=True)

    def __len__(self):
        return len(self.min_heap) + len(self.max_heap)

    def add(self, x):
        left = max(int(self.p * len(self)), 1)

        if left > len(self.max_heap):
            self.max_heap.push(x)
        else:
            self.min_heap.push(x)

        if len(self.min_heap) > 0 and self.min_heap.top() < self.max_heap.top():
            max_top = self.max_heap.pop()
            min_top = self.min_heap.pop()
            self.min_heap.push(max_top)
            self.max_heap.push(min_top)
        
        return self.get()

    def get(self):
        return self.max_heap.top()
