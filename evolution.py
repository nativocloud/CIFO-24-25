import random
from operators import (
    mutate_swap,
    mutate_team_shift,
    mutate_shuffle_team,
    crossover_one_point,
    crossover_uniform,
    selection_tournament,
    selection_ranking
)
from solution import LeagueSolution

# Map operator names to functions for flexible configuration
MUTATIONS = [mutate_swap, mutate_team_shift, mutate_shuffle_team]
CROSSOVERS = [crossover_one_point, crossover_uniform]
SELECTIONS = [selection_tournament, selection_ranking]

def generate_population(players, size):
    population = []
    while len(population) < size:
        candidate = LeagueSolution()
        if candidate.is_valid(players):
            population.append(candidate)
    return population

def genetic_algorithm(
    players,
    population_size=50,
    generations=30,
    mutation_rate=0.2,
    elite_size=5,
    mutation_operator=mutate_swap,
    crossover_operator=crossover_one_point,
    selection_operator=selection_tournament
):
    population = generate_population(players, population_size)
    history = []
    best_solution = min(population, key=lambda s: s.fitness(players))

    for gen in range(generations):
        new_population = []
        population.sort(key=lambda x: x.fitness(players))
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            # No loop principal do GA
            parent1 = selection_operator(population, players)
            parent2 = selection_operator(population, players)
            
            #parent1 = selection_operator(population)
            #parent2 = selection_operator(population)
            child = crossover_operator(parent1, parent2)

            if random.random() < mutation_rate:
                child = mutation_operator(child)

            if child.is_valid(players):
                new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda s: s.fitness(players))
        if current_best.fitness(players) < best_solution.fitness(players):
            best_solution = current_best
        history.append(best_solution.fitness(players))

    return best_solution, history
