# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:58:07 2023

@author: mcamboim
"""

import numpy as np


class GeneticAlgorithm:
    
    def __init__(self,partition):
        self.__partition = partition
        self.__individual_size = len(self.__partition)
    
    def popInit(self,pop_size):
        individual_to_be_shuffled = np.zeros(300)
        individual_to_be_shuffled[:150] = 1 
        self.__pop = np.zeros((pop_size,self.individual_size))
        for line_idx in range(pop_size):
            init = np.random.randint(-5,6)
            individual_to_be_shuffled = np.zeros(300)
            individual_to_be_shuffled[:(init+150)] = 1 
            np.random.shuffle(individual_to_be_shuffled)
            self.__pop[line_idx,:] = individual_to_be_shuffled
        self.__pop_fitness = np.zeros(pop_size)
      
    def runGa(self,pop_size,crossover_probability,mutation_probability,elitism_percentage,generations):
        # Begin Variables
        self.__pop_worst_objective_by_generation = np.zeros(generations)
        self.__pop_best_objective_by_generation = np.zeros(generations)
        self.__pop_mean_objective_by_generation = np.zeros(generations)
        # 1. Initialization of the population
        self.popInit(pop_size)
        # 2. Get Fitness
        self.__pop_fitness = self.getPopFitness(self.population)
        # Run Genetic Algorithm
        for generation in range(generations):
            print(f'{generation+1}/{generations} -> {self.best_objective_function}')
            self.__pop_worst_objective_by_generation[generation] = self.worst_objective_function
            self.__pop_best_objective_by_generation[generation] = self.best_objective_function
            self.__pop_mean_objective_by_generation[generation] = self.mean_objective_function
            # Local Search
            if(generation % 5 == 0):
                self.local_search()            
            # Selection
            #pop_selected = self.tournamentSelection(pop_size,pop_size,11)
            pop_selected = self.rouletteWheelSelection(pop_size)
            # Crossover
            pop_son = self.crossoverForBinary(pop_selected,crossover_probability)
            # Mutation
            pop_son = self.mutationForBinary(pop_son,mutation_probability)
            pop_son_fitness = self.getPopFitness(pop_son)
            # Elitism
            self.elitism(pop_son, pop_son_fitness, elitism_percentage)
            self.__pop_fitness = self.getPopFitness(self.population)
            
    # Fitness ================================================================
    def getObjectiveFunction(self,individual):
        sum_subset_A = -np.sum((individual-1) * self.partition) # 0 is A
        sum_subset_B = np.sum((individual) * self.partition)    # 1 is B
        objective_function_value = np.abs(sum_subset_A - sum_subset_B)
        return objective_function_value
    
    def getPopFitness(self,pop):
        pop_size = pop.shape[0]
        pop_fitness = np.zeros(pop_size)
        for this_individual_idx in range(pop_size):
            objective_function_value = self.getObjectiveFunction(pop[this_individual_idx,:])
            pop_fitness[this_individual_idx] = 1 / (1 + objective_function_value)
        return pop_fitness
    
    # Selection ==============================================================    
    def rouletteWheelSelection(self,individuals_to_be_selected):
        pop_selected = np.zeros((individuals_to_be_selected,self.individual_size))
        cum_probability = np.cumsum(self.fitness) / np.sum(self.fitness)
        for pop_selected_idx in range(individuals_to_be_selected):
            random_uniform_number = np.random.uniform(0,1.0)
            selected_individual_idx = np.argwhere(cum_probability > random_uniform_number)[0,0]
            pop_selected[pop_selected_idx,:] = self.population[selected_individual_idx,:]
        return pop_selected
    
    # Crossover ==============================================================
    def crossoverForBinary(self,pop_selected,crossover_probability):
        #pop_son = np.zeros((pop_selected.shape[0],19))
        pop_son = np.copy(pop_selected)
        # Get selected individuals to perform crossover
        individuals_to_crossover = []       
        for pop_selected_idx in range(pop_son.shape[0]):
            random_uniform_number = np.random.uniform(0,1.0)
            if(random_uniform_number < crossover_probability):
                individuals_to_crossover.append(pop_selected_idx)
        # The number of individuals to perform crossover should be even
        if(len(individuals_to_crossover) % 2 == 1):
            individuals_to_crossover.pop()
        # Now perform the crossover in the selected individuals
        for pop_crossover_idx in range(0,len(individuals_to_crossover),2):
            individual_to_crossover_idx1 = individuals_to_crossover[pop_crossover_idx]
            individual_to_crossover_idx2 = individuals_to_crossover[pop_crossover_idx+1]
            individual_to_crossover_1 = np.copy(pop_selected[individual_to_crossover_idx1])
            individual_to_crossover_2 = np.copy(pop_selected[individual_to_crossover_idx2])
            
            son_1,son_2 = self.crossover(individual_to_crossover_1, individual_to_crossover_2)
            pop_son[individual_to_crossover_idx1,:] = np.copy(son_1)
            pop_son[individual_to_crossover_idx2,:] = np.copy(son_2)
        
        return pop_son
    
    def crossover(self,father_1,father_2):
        son_1 = np.zeros(self.individual_size)
        son_2 = np.zeros(self.individual_size)
        
        break_point = np.random.randint(0,self.individual_size)
        son_1[:break_point] = father_1[:break_point]
        son_1[break_point:] = father_2[break_point:]
        son_2[:break_point] = father_2[:break_point]
        son_2[break_point:] = father_1[break_point:]
        
        return son_1,son_2        
    
    # Mutation ===============================================================
    def mutationForBinary(self,pop_son,mutation_probability):
        for pop_son_idx in range(pop_son.shape[0]):
            for pop_value_idx in range(self.individual_size):
                random_uniform_number = np.random.uniform(0,1.0)
                if(random_uniform_number < mutation_probability):
                    if(pop_son[pop_son_idx,pop_value_idx]==0):
                        pop_son[pop_son_idx,pop_value_idx]=1
                    else:
                        pop_son[pop_son_idx,pop_value_idx]=0
        return pop_son
    
    # Elitism ================================================================
    def elitism(self,pop_son,pop_son_fitness,elitism_percentage):
        pop_size = self.population.shape[0]
        individuals_to_be_replaced = int(pop_size * elitism_percentage)
        ordered_fathers_idx = np.argsort(self.fitness)
        ordered_fathers = np.copy(self.population[ordered_fathers_idx,:]) # Worst to best
        ordered_sons_idx = np.flip(np.argsort(pop_son_fitness)) # Best to worst
        ordered_sons = np.copy(pop_son[ordered_sons_idx,:])
        ordered_fathers[0:individuals_to_be_replaced] = np.copy(ordered_sons[0:individuals_to_be_replaced]) 
        self.__pop = np.copy(ordered_fathers)
        
    # Local Search ===========================================================
    def local_search(self):
        best_individual_idx = np.argmax(self.fitness)
        best_individual = np.copy(self.population[best_individual_idx])
        print(self.fitness[best_individual_idx])
        for idx in range(self.individual_size):
            best_individual_local = np.copy(best_individual)
            previous_fitness = self.getPopFitness(best_individual_local.reshape(1,-1))
            if(best_individual_local[idx] == 0):
                best_individual_local[idx] = 1
            else:
                best_individual_local[idx] = 0
            new_fitness = self.getPopFitness(best_individual_local.reshape(1,-1))[0]
            if(new_fitness > previous_fitness):
                best_individual = np.copy(best_individual_local)
        
        self.__pop[best_individual_idx] = np.copy(best_individual)
        self.__pop_fitness[best_individual_idx] = self.getPopFitness(best_individual.reshape(1,-1))
        print(self.fitness[best_individual_idx])
        
    # Properties =============================================================
    @property
    def population(self):
        return self.__pop
    
    @property
    def partition(self):
        return self.__partition
    
    @property
    def individual_size(self):
        return self.__individual_size
    
    @property
    def fitness(self):
        return self.__pop_fitness
    
    @property
    def best_fitness(self):
        return np.max(self.__pop_fitness)
    
    @property
    def worst_objective_through_generations(self):
        return self.__pop_worst_objective_by_generation
    
    @property
    def best_objective_through_generations(self):
        return self.__pop_best_objective_by_generation
    
    @property
    def mean_objective_through_generations(self):
        return self.__pop_mean_objective_by_generation
    
    @property
    def best_objective_function(self):
        best_fitness_idx = np.argmax(self.fitness)
        best_fitness_objetctive_function = self.getObjectiveFunction(self.population[best_fitness_idx,:])
        return best_fitness_objetctive_function
    
    @property
    def worst_objective_function(self):
        worst_fitness_idx = np.argmin(self.fitness)
        worst_fitness_objetctive_function = self.getObjectiveFunction(self.population[worst_fitness_idx,:])
        return worst_fitness_objetctive_function
    
    @property
    def mean_objective_function(self):
        pop_size = self.population.shape[0]
        fitnesses = np.zeros(pop_size)
        for individual_idx in range(pop_size):
            fitnesses[individual_idx] = self.getObjectiveFunction(self.population[individual_idx,:])
        return np.mean(fitnesses)
    
        
            