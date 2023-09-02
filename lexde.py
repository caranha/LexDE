# Adapted LexDE by Claus Aranha
import numpy as np
import csv
import tqdm

# TODO: understand this implementation of polynomial mutation
# TODO: Evolution object with restart

### Problem Definition ###

def standard_fun1(x):
    return np.sum(x*x)

def standard_fun2(x):
    return np.sum(x*x-1)

class Individual:
    def __init__(self, parameters):
        self.parameters = parameters
        self.fitness = None

class Problem:
    def __init__(self, n_var = 10, xl = -5., xu = 5.,
                 problems = None, logfile = None):
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.logfile = logfile
        self.n_eval = 0

        if problems:
            self.problems = problems
        else:
            self.problems = [standard_fun1, standard_fun2]

    def generate_population(self, npop):
        _pop = [Individual(np.random.uniform(self.xl,
                                             self.xu,
                                             self.n_var)) for _ in range(npop)]
        return _pop

    def evaluate_individual(self, individual):
        individual.fitness = [fun(individual.parameters) for fun in self.problems]
        return individual
    
    def evaluate_population(self, pop):

        # You can parallelize here
        population = list(map(self.evaluate_individual, pop))

        # log individuals
        if self.logfile:
            with open(self.logfile, "a", newline="") as log:
                writer = csv.writer(log)
                for p in population:
                    output = [self.n_eval] + list(p.parameters) + list(p.fitness)
                    writer.writerow(output)
                    self.n_eval += 1
        else:
            self.n_eval += len(population)
                    


### LEXICASE SELECTION ###

# Used to define sigma in auto epsilon lexicase selection (eq 2)
def median_absolute_deviation(fitness_list):
    median_val = np.median(fitness_list)
    mad = np.median([abs(x - median_val) for x in fitness_list])
    return mad

# Lexicase selection (alg 1)
def auto_eps_lex(population): 
    candidates = population
    cases = list(range(len(population[0].fitness))) # fitness index
    np.random.shuffle(cases)                         # shuffle objectives

    while len(cases) > 0 and len(candidates) > 1:
        fitness_list = [x.fitness[cases[0]] for x in candidates]
        mad = median_absolute_deviation(fitness_list)
        best = min(fitness_list)
        candidates = [x for x in candidates if x.fitness[cases[0]] <= best + mad]
        cases.pop(0)

    return np.random.choice(candidates)

### Differential Evolution ###

# TODO: Understand this part better - not listed in paper! :-(
def poly_mut(x, xl, xu, eta = 20):
    size = len(x)

    delta_1 = (x - xl) / (xu - xl)
    delta_2 = (xu - x) / (xu - xl)
    mut_pow = 1. / (eta + 1.)

    mu = np.random.random(size)
    xy = 1. - delta_1
    val = 2. * mu + (1. - 2. * mu) * xy ** (eta + 1)
    delta_q = val ** mut_pow - 1.

    xy = 1. - delta_2
    val = 2. * (1. - mu) + 2. * (mu - .5) * xy ** (eta + 1)
    delta_q_ = 1. - val ** mut_pow

    delta_q[mu >= .5] = delta_q_[mu >= .5]

    x = x + delta_q * (xu - xl)
    return x

def create_offspring(population, problem, CR = .5, F = .5, eta = 20):
    # lexicase selection for 1st parent
    x_lex = auto_eps_lex(population).parameters
    
    # random selection for 2nd, 3rd parent
    n_pop = len(population)
    xr1 = population[np.random.randint(n_pop)].parameters
    xr2 = population[np.random.randint(n_pop)].parameters

    # differential mutation
    yi = x_lex + F * (xr1 - xr2)
    
    # binomial crossover
    n_var = len(x_lex)
    maskbits = np.random.random(n_var)
    maskbits[np.random.randint(n_var)] = 0
    yi[maskbits < CR] = x_lex[maskbits < CR]

    # fix bounds before polynomial mutation
    yi = np.clip(yi, problem.xl, problem.xu)
    
    # polynomial mutation
    yi = poly_mut(yi, problem.xl, problem.xu, eta)
    yi = np.clip(yi, problem.xl, problem.xu)
    
    yi = Individual(yi)
    return yi
    

# example optimization
