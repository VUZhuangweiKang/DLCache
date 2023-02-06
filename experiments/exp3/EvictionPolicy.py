import random
from collections import defaultdict
from collections import OrderedDict


class LFUCache:
    def __init__(self, capacity: float):
        self.cache = {}
        self.frequency = defaultdict(OrderedDict)
        self.min_frequency = 1
        self.capacity = capacity

    def get(self, key: str, current_t: float=None) -> float:
        if key not in self.cache:
            return -1
        value = self.cache[key]
        frequency = value[1]
        self.frequency[frequency].pop(key)
        if frequency == self.min_frequency and not self.frequency[frequency]:
            self.min_frequency += 1
        value[1] += 1
        self.frequency[value[1]][key] = value
        return value[0]

    def put(self, key: str, value: float, current_t: float=None) -> None:
        if self.capacity == 0:
            return
        if key in self.cache:
            self.get(key)
        else:
            while True:
                total = 0
                for v in self.cache.values():
                    total += v[0]
                if total + value >= self.capacity:
                    for i in range(self.min_frequency, 10000):
                        if self.frequency[i]:
                            evicted_key = next(iter(self.frequency[i].keys()))
                            self.frequency[i].pop(evicted_key)
                            self.cache.pop(evicted_key)
                            break
                else:
                    break
            self.cache[key] = [value, 1]
            self.frequency[1][key] = self.cache[key]
            self.min_frequency = 1


class LRUCache:
    def __init__(self, capacity: float):
        self.cache = {}
        self.capacity = capacity
        self.lru = []

    def get(self, key: str, current_t: float=None) -> float:
        if key not in self.cache:
            return -1
        self.lru.remove(key)
        self.lru.append(key)
        return self.cache[key]

    def put(self, key: str, value: float, current_t: float=None) -> None:
        if key in self.cache:
            self.lru.remove(key)
        else:
            while sum(self.cache.values()) + value > self.capacity:
                del self.cache[self.lru[0]]
                self.lru.pop(0)
        self.cache[key] = value
        self.lru.append(key)


class LRFUCache:
    def __init__(self, capacity: float, attenuation=2.0, step=0.5):
        self.cache = {}
        self.access_time = defaultdict(list)
        self.capacity = capacity
        self.attenuation = attenuation
        self.step = step

    def get(self, key: str, current_t: float) -> float:
        if key not in self.cache:
            return -1
        self.access_time[key].append(current_t)
        return self.cache[key]
    
    def put(self, key: str, value: float, current_t: float):
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key] = value
            self.access_time[key].append(current_t)
        else:
            scores = []
            for etag in self.cache:
                crf = 0
                for at in self.access_time[etag]:
                    dur = current_t - at
                    crf += pow(1/self.attenuation, self.step*dur)
                
                if crf > 0:
                    scores.append((etag, crf))
            scores.sort(key=lambda x: x[1])
            
            i = 0
            while sum(self.cache.values()) + value > self.capacity:
                self.cache.pop(scores[i][0])
                i += 1
            self.cache[key] = value
            self.access_time[key].append(current_t)
            
            
class RandomCache:
    def __init__(self, capacity: float):
        self.cache = {}
        self.capacity = capacity

    def get(self, key: str, current_t: float=None) -> float:
        if key not in self.cache:
            return -1
        return self.cache[key]

    def put(self, key: str, value: float, current_t: float=None) -> None:
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key] = value
        else:
            while sum(self.cache.values()) + value > self.capacity:
                random_key = random.choice(list(self.cache.keys()))
                self.cache.pop(random_key)
            self.cache[key] = value


class CostAwareGreedyCache:
    def __init__(self, capacity: float, costs: dict):
        self.cache = {}
        self.capacity = capacity
        self.costs = costs

    def get(self, key: str, current_t: float) -> float:
        if key not in self.cache:
            return -1
        return self.cache[key]
    
    def put(self, key: str, value: float, current_t: float):
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key] = value
        else:
            scores = []
            for etag in self.cache:
                scores.append((etag, self.costs[etag]))

            scores.sort(key=lambda x: x[1])
            i = 0
            while sum(self.cache.values()) + value > self.capacity:
                self.cache.pop(scores[i][0])
                i += 1
            self.cache[key] = value
            
  
class CostAwareLRFUCache:
    def __init__(self, capacity: float, costs: dict, attenuation=2.0, step=0.5):
        self.cache = {}
        self.access_time = defaultdict(list)
        self.capacity = capacity
        self.costs = costs
        self.attenuation = attenuation
        self.step = step

    def get(self, key: str, current_t: float) -> float:
        if key not in self.cache:
            return -1
        self.access_time[key].append(current_t)
        return self.cache[key]
    
    def put(self, key: str, value: float, current_t: float):
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key] = value
            self.access_time[key].append(current_t)
        else:
            scores = []
            for etag in self.cache:
                crf = 0
                for at in self.access_time[etag]:
                    dur = current_t - at
                    crf += pow(1/self.attenuation, self.step*dur)

                if crf != 0:
                    scores.append((etag, crf, self.costs[etag]))

            scores.sort(key=lambda x: (x[1], x[2]))
            i = 0
            while sum(self.cache.values()) + value > self.capacity:
                self.cache.pop(scores[i][0])
                i += 1

            self.cache[key] = value
            self.access_time[key].append(current_t)