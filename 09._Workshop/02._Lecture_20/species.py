import random
import string
from typing import Protocol
from Levenshtein import distance as lev_dist


class Specie(Protocol):
    def __init__(self, genome_len: (int|None)=None, DNA: (str|None)=None, delegate=None):
        pass

    def __str__(self) -> str:
        pass

    def fitness(self, other: 'Specie') -> float:
        pass


class StringDNASpecie(Specie):
    alphabet = list(string.ascii_letters)
    alphabet.append(' ')

    def __init__(self, genome_len: (int|None)=None, DNA: (str|None)=None, delegate=None):
        if DNA:
            self.DNA = DNA
        else:
            if not genome_len: genome_len = 8
            self.DNA = ''.join(random.choice(self.alphabet) for i in range(genome_len))
        if delegate:
            self.delegate = delegate

    def __str__(self):
        return self.DNA

    def __iter__(self):
        return iter(self.DNA)

    def __getitem__(self, index):
        return self.DNA[index]

    def __len__(self):
        return len(self.DNA)

    def __lt__(self, other: Specie) -> bool:
        return self.fitness(self.delegate.target) < other.fitness(other.delegate.target)

    def __eq__(self, other: Specie) -> bool:
        return self.DNA == other.DNA

    def fitness(self, other: Specie) -> float:
        dist = lev_dist(self.DNA, other.DNA)
        if dist > 0:
            return 1.0/((dist+1)**2)
        else:
            return 1.0

    def mutate(self, mutation_rate) -> None:
        if random.random() < mutation_rate:
            index = random.randint(0, len(self.DNA) - 1)
            self.DNA = self.DNA[0:index] + random.choice(self.alphabet) + self.DNA[index:-1]

    @classmethod
    def copulate(cls, parents) -> Specie:
        baby_DNA = []
        for i, j in zip(parents[0], parents[1]):
            if random.random() > 0.5:
                baby_DNA.append(i)
            else:
                baby_DNA.append(j)
        return cls(DNA=''.join(baby_DNA))



class Population:
    def __init__(self, *, population_size: int, target: Specie):
        self.mutation_rate = 0.02
        self.target = target
        self.population_size = population_size
        self.population = [type(target)(genome_len=len(target)) for i in range(population_size)]
        # setting the delegate during init of collection items for some reason not possible
        for i in self.population: i.delegate = self

    def __iter__(self):
        return iter(self.population)

    def __getitem__(self, index):
        return self.population[index]

    def __len__(self):
        return len(self.population)

    def mutate(self) -> None:
        for i in self.population:
            i.mutate(mutation_rate=self.mutation_rate)

    def debug(self) -> None:
        for i in sorted(self.population):
            print(i.DNA, i.fitness(self.target))

    def selection(self) -> None:
        self.population = sorted(self.population)
        self.population = self.population[len(self.population)//2:]
        while len(self.population) < self.population_size:
            parents = random.sample(self.population, 2)
            baby = type(self.target).copulate(parents)
            baby.delegate = self
            self.population.append(baby)

    def best_fit(self) -> float:
        best = sorted(self.population)[-1]
        print('..', best.DNA, best.fitness(self.target))
        return best.fitness(self.target)