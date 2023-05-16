# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:58:07 2023

@author: mcamboim
"""

import numpy as np


class GeneticAlgorithm:
    
    def __init__(self,partition):
        self.__partition = np.append(partition,-1)
        self.__individual_size = len(self.__partition)
    
    def popInit(self):
        values_to_shuffle = np.copy(self.__partition)
        for individual_idx in range(self.__pop_size):
            np.random.shuffle(values_to_shuffle)
            self.__pop[individual_idx,:] = values_to_shuffle
            
    def runGa(self,pop_size,crossover_probability,mutation_probability,elitism_percentage,generations):
        # Begin Variables
        self.__pop = np.zeros((pop_size,self.__individual_size))
        self.__pop_best_objective_by_generation = np.zeros(generations)
        self.__pop_mean_objective_by_generation = np.zeros(generations)
        self.__pop_size = pop_size
        
        # 1. Initialization of the population
        self.popInit()
        # 2. Get Fitness
        self.__pop_fitness = self.getPopFitness(self.population)
        # Run Genetic Algorithm
        for generation in range(generations):
            print(f'{generation+1}/{generations} -> {self.best_objective_function}')
            self.__pop_best_objective_by_generation[generation] = self.best_objective_function
            self.__pop_mean_objective_by_generation[generation] = self.mean_objective_function
            
            pop_selected = self.tournamentSelection(pop_size)
            pop_son = self.crossoverForPermutation(pop_selected,crossover_probability)
            pop_son = self.mutationForPermutation(pop_son,mutation_probability)
            pop_son_fitness = self.getPopFitness(pop_son)
            self.elitism(pop_son, pop_son_fitness, elitism_percentage)
            self.__pop_fitness = self.getPopFitness(self.population)
            
    # Fitness ================================================================
    def getObjectiveFunction(self,individual):
        breakpoint_idx = np.argwhere(individual==-1)[0,0]
        subset_A = individual[:breakpoint_idx]
        subset_B = individual[breakpoint_idx+1:]
        objective_function_value = np.abs(np.sum(subset_A) - np.sum(subset_B))
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
        pop_selected = np.zeros((individuals_to_be_selected,self.__individual_size))
        cum_probability = np.cumsum(self.fitness) / np.sum(self.fitness)
        for pop_selected_idx in range(individuals_to_be_selected):
            random_uniform_number = np.random.uniform(0,1.0)
            selected_individual_idx = np.argwhere(cum_probability > random_uniform_number)[0,0]
            pop_selected[pop_selected_idx,:] = self.population[selected_individual_idx,:]
        return pop_selected
    
    def tournamentSelection(self,individuals_to_be_selected):
        pop_selected = np.zeros((individuals_to_be_selected,self.__individual_size))
        for pop_selected_idx in range(individuals_to_be_selected):
            selected_individuals = np.random.randint(0,self.__pop_size,3)
            winner_idx = np.argmax(self.fitness[selected_individuals])
            selected_individual_idx = selected_individuals[winner_idx]
            pop_selected[pop_selected_idx,:] = self.population[selected_individual_idx,:]
        return pop_selected
    
    # Crossover ==============================================================
    def crossoverForPermutation(self,pop_selected,crossover_probability):
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
            
            son_1,son_2 = self.crossoverOX(individual_to_crossover_1, individual_to_crossover_2)
            pop_son[individual_to_crossover_idx1,:] = np.copy(son_1)
            pop_son[individual_to_crossover_idx2,:] = np.copy(son_2)
        
        return pop_son
    
    def crossoverOX(self,individual_to_crossover_1,individual_to_crossover_2):
        son_1 = np.zeros(self.__individual_size) - 1
        son_2 = np.zeros(self.__individual_size) - 1
        
        break_point_1 = np.random.randint(0,self.__individual_size)
        break_point_2 = np.random.randint(0,self.__individual_size)
        while(break_point_2 == break_point_1):
            break_point_2 = np.random.randint(0,self.__individual_size)
        if(break_point_1 > break_point_2):
            break_point_1,break_point_2 = break_point_2,break_point_1
            
        son_1[break_point_1:break_point_2] = np.copy(individual_to_crossover_1[break_point_1:break_point_2])
        son_2[break_point_1:break_point_2] = np.copy(individual_to_crossover_2[break_point_1:break_point_2])
        
        indexes_after_break_point = np.roll(np.arange(self.__individual_size),-break_point_2)
        position_to_be_replaced_idx_1 = 0
        position_to_be_replaced_idx_2 = 0
        for indexes_after_break_point_value in indexes_after_break_point:
            individual_to_crossover_1_value_by_idx = individual_to_crossover_1[indexes_after_break_point_value]
            individual_to_crossover_2_value_by_idx = individual_to_crossover_2[indexes_after_break_point_value]
            # Son 1
            if(individual_to_crossover_2_value_by_idx not in son_1):
                position_to_be_replaced = indexes_after_break_point[position_to_be_replaced_idx_1]
                son_1[position_to_be_replaced] = individual_to_crossover_2_value_by_idx
                position_to_be_replaced_idx_1 =  position_to_be_replaced_idx_1 + 1
            # Son 2
            if(individual_to_crossover_1_value_by_idx not in son_2):
                position_to_be_replaced = indexes_after_break_point[position_to_be_replaced_idx_2]
                son_2[position_to_be_replaced] = individual_to_crossover_1_value_by_idx
                position_to_be_replaced_idx_2 =  position_to_be_replaced_idx_2 + 1
        
        return son_1,son_2        
    
    # Mutation ===============================================================
    def mutationForPermutation(self,pop_son,mutation_probability):
        for pop_son_idx in range(pop_son.shape[0]):
            random_uniform_number = np.random.uniform(0,1.0)
            if(random_uniform_number < mutation_probability):
                break_point = np.argwhere(pop_son[pop_son_idx,:]==-1)[0,0]
                break_point_1 = np.random.randint(0,break_point+1)
                break_point_2 = np.random.randint(break_point,self.__individual_size)
                #while(break_point_2 == break_point_1):
                #    break_point_2 = np.random.randint(break_point,self.__individual_size)
                pop_son[pop_son_idx,break_point_1],pop_son[pop_son_idx,break_point_2] =  pop_son[pop_son_idx,break_point_2],pop_son[pop_son_idx,break_point_1]
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
    
    # Properties =============================================================
    @property
    def population(self):
        return self.__pop
    
    @property
    def fitness(self):
        return self.__pop_fitness
    
    @property
    def best_fitness(self):
        return np.max(self.__pop_fitness)
    
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
    def mean_objective_function(self):
        pop_size = self.population.shape[0]
        fitnesses = np.zeros(pop_size)
        for individual_idx in range(pop_size):
            fitnesses[individual_idx] = self.getObjectiveFunction(self.population[individual_idx,:])
        return np.mean(fitnesses)
    
        
            