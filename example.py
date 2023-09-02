import tqdm
import numpy as np
from lexde import Problem, Individual, create_offspring

# Task definition. Replace this with your own tasks
def standard_fun1(x):
    return np.sum(x*x)

def standard_fun2(x):
    return np.sum(x*x-1)

tasks = [standard_fun1, standard_fun2]

# Running the optimization
if __name__ == "__main__":

    max_evals = 5000
    n_pop = 20
    
    F = .5
    CR = .5
    eta = 20

    pbar = tqdm.tqdm(total = max_evals)

    # Initialize problem and population:
    problem = Problem(n_var = 10, xl = -5., xu = 5.,
                      problems = tasks,
                      logfile = "sample_log.csv")
    population = problem.generate_population(n_pop)

    problem.evaluate_population(population)
    pbar.update(len(population))

    while (problem.n_eval < max_evals):
        offspring = []
        for i, xi in enumerate(population):
            yi = create_offspring(population, problem,
                                  CR, F, eta)
            offspring.append(yi)
        
        population = offspring

        # Evaluate new population
        problem.evaluate_population(population)
        pbar.update(n_pop)

