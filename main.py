# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:45:26 2023

@author: mcamboim
"""
from genetic_algorithm_local_search import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close('all')
plt.rcParams['axes.linewidth'] = 2.0
plt.rc('axes', axisbelow=True)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

pop_size = 500
crossover_probability = 0.8
mutation_probability = 0.02
generations = 1000
elitism_percentage = 0.1   
executions = 10

partition = pd.read_csv('particao.txt',header=None).values.reshape(-1)

gen_alg = GeneticAlgorithm(partition)
best_objective = np.zeros((generations,executions))
worst_objective = np.zeros((generations,executions))
mean_objective = np.zeros((generations,executions))
for execution_idx in range(executions):
    print(f'\nExecução {execution_idx+1}/{executions}')
    gen_alg.runGa(pop_size=pop_size,crossover_probability=crossover_probability,mutation_probability=mutation_probability,elitism_percentage=elitism_percentage,generations=generations)
    best_objective[:,execution_idx] = gen_alg.best_objective_through_generations
    worst_objective[:,execution_idx] = gen_alg.worst_objective_through_generations
    mean_objective[:,execution_idx] = gen_alg.mean_objective_through_generations

x_mean = np.array([np.mean(best_objective[generation,:]) for generation in range(generations)])
x_std = np.array([np.std(best_objective[generation,:],ddof=1) for generation in range(generations)])
plt.figure(figsize=(12,8),dpi=150)
plt.subplot(2,1,1)
plt.plot(x_mean,lw=2,c='b')
#plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
plt.fill_between(np.arange(generations),x_mean-1*x_std,x_mean+x_std,alpha=0.5)
plt.ylabel('Função Objetivo')
plt.xlabel('(a) # de Gerações [n]')
plt.legend(['Valor Médio da Evolução da Função Objetivo','Desvio Padrão da Evolução da Função Objetivo'])
plt.xlim([0,generations-1])
plt.grid(True,ls='dotted')

plt.subplot(2,1,2)
plt.plot(x_mean,lw=2,c='b')
#plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
plt.fill_between(np.arange(generations),x_mean-1*x_std,x_mean+x_std,alpha=0.5)
plt.ylabel('Função Objetivo')
plt.xlabel('(b) # de Gerações [n]')
plt.legend(['Valor Médio da Evolução da Função Objetivo','Desvio Padrão da Evolução da Função Objetivo'])
plt.xlim([0,generations-1])
plt.grid(True,ls='dotted')
plt.ylim([-2000.0,2000.0])
plt.tight_layout()

plt.figure(figsize=(12,8),dpi=150)
for line_idx in range(2):
    for column_idx in range(5):
        plt.subplot(2,5,line_idx*5+column_idx+1)
        plt.plot(best_objective[:,line_idx*5+column_idx],c='b')
        plt.plot(worst_objective[:,line_idx*5+column_idx],c='r')
        plt.plot(mean_objective[:,line_idx*5+column_idx],c='g')
        plt.ylim([-5e5,1e7])
        if(column_idx == 0):
            plt.ylabel('Função Objetivo')
        if(line_idx==1):
            plt.xlabel('# de Gerações [N]')
        if(line_idx==0 and column_idx == 0):
            plt.legend(['B','W','M'])
        plt.grid(True,ls='dotted')
        plt.xlim([0,1e3])
plt.tight_layout()
