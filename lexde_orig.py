# Original LexDE by Yifan He
import numpy as np

class Problem:

    def __init__(self, n_var=10, xl=-5, xu=5, record=None):
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.record = record
        self.n_eval = 0

        if self.record:
            with open(self.record, "w") as log:
                log.close()

    def evaluate(self, individual):
        individual.fitness = self.function(individual.genome)
        self.n_eval += 1

        if self.record:
            with open(self.record, "a") as log:
                s = ""
                s += str(self.n_eval) + ","
                s += ",".join([str(i) for i in individual.genome]) + ","
                s += ",".join(str(i) for i in individual.fitness) + "\n"
                log.write(s)

        return individual

    # fitness function of the problem
    def function(self, x):
        f1 = np.sum(x*x)
        f2 = np.sum(x*x-x)
        return (f1, f2)


class Individual:

    def __init__(self, genome):
        self.genome = genome
        self.fitness = None


def auto_eps_lex(population):
    candidates = population
    cases = list(range(len(population[0].fitness)))
    np.random.shuffle(cases)

    while len(cases) > 0 and len(candidates) > 1:
        errors_for_this_case = [x.fitness[cases[0]] for x in candidates]
        median_val = np.median(errors_for_this_case)
        median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
        best_val_for_case = min(errors_for_this_case)
        max_val_to_survive = best_val_for_case + median_absolute_deviation
        candidates = [x for x in candidates if x.fitness[cases[0]] <= max_val_to_survive]

        cases.pop(0)

    return np.random.choice(candidates)


def poly_mut(x, xl, xu, eta=20):
    size = len(x)

    delta_1 = (x - xl) / (xu - xl)
    delta_2 = (xu - x) / (xu - xl)
    mut_pow = 1. / (eta + 1.)

    mu = np.random.random(size)
    xy = 1. - delta_1
    val = 2. * mu + (1.-2.*mu) * xy ** (eta+1)
    delta_q = val ** mut_pow - 1.

    xy = 1. - delta_2
    val = 2. * (1.-mu) + 2. * (mu-.5) * xy ** (eta+1)
    delta_q_ = 1. - val ** mut_pow

    delta_q[mu>=.5] = delta_q_[mu>=.5]

    x = x + delta_q * (xu - xl)
    return x


n_var = 10
xl, xu = -5., 5.
problem = Problem(n_var, xl, xu, record="log.csv")

n_pop = 20
n_gen = 50
F = .5
CR = .5
eta = 20

# initialize a population
population = [Individual(np.random.uniform(xl,xu,n_var)) for _ in range(n_pop)]
# evaluate fitnesses
population = list(map(problem.evaluate, population))

for g in range(1, n_gen):

    offspring = []

    for i, xi in enumerate(population):

        # lexicase selection for 1st parent
        x_lex = auto_eps_lex(population).genome

        # random selection for rest parents
        xr1 = population[np.random.randint(n_pop)].genome
        xr2 = population[np.random.randint(n_pop)].genome

        # differential mutation
        yi = x_lex+F*(xr1-xr2)

        # binomial crossover
        maskbits = np.random.random(n_var)
        maskbits[np.random.randint(n_var)] = 0
        yi[maskbits<CR] = x_lex[maskbits<CR]

        # fix bounds before polynomial mutation
        yi = np.clip(yi, xl, xu)

        # polynomial mutation
        yi = poly_mut(yi, xl, xu, eta)

        # fix bounds
        yi = np.clip(yi, xl, xu)

        yi = Individual(yi)
        offspring.append(yi)

    population = offspring
    # evaluate fitnesses
    population = list(map(problem.evaluate, population))
