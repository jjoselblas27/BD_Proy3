import heapq

class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, element):
        # Multiplicamos la distancia por -1 para convertir la heap en una max heap
        heapq.heappush(self.heap, [-element[0], element[1]])

    def pop(self):
        # Al extraer un elemento, multiplicamos nuevamente por -1 para obtener la distancia original
        popped_element = heapq.heappop(self.heap)
        return [-popped_element[0], popped_element[1]]

    def peek(self):
        # Devuelve el elemento máximo sin extraerlo
        if self.heap:
            return [-self.heap[0][0], self.heap[0][1]]
  
    def __len__(self):
        return len(self.heap)

