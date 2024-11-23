# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2022

@author: Thaer

@Customized version of EvoloPy to fit generelized island model


"""
from pathlib import Path

# -------------import UDA Algorithms-------------------------
import My_Algorithms.HGS_UDA as hgs
import My_Algorithms.EHGS_UDA as ehgs
import My_Algorithms.HHO_UDA as hho
import My_Algorithms.BAT_UDA as bat
import My_Algorithms.SCA_UDA as sca
import math_functions_details as mf_details       #benchmarks

# -------------Others----------------------------------------
import math
import csv
import numpy as np
import time
import warnings
import os
import plot_convergence as conv_plot
import plot_boxplot as box_plot
import ast

from solution import solution

# Mathematical Evaluation Functions
import My_Problems.Math_Benchmarks as Benchmarks
import My_Problems.Math_Benchmarks_CEC2017 as Benchmarks_CEC2017


# Pygmo library
import pygmo as pg


warnings.simplefilter(action="ignore")


def selector(algorithm, pop, Iter, function_name, Export_diversity):   

    # --------- UDA algorithms ------------------------
    
    if algorithm == "EHGS_UDA":
        algo = pg.algorithm(ehgs.my_EHGS(gen = Iter))
        
    elif algorithm == "HGS_UDA":
        algo = pg.algorithm(hgs.my_HGS(gen = Iter))
       
    elif algorithm == "HHO_UDA":
        algo = pg.algorithm(hho.my_HHO(gen = Iter))
    
    elif algorithm == "BAT_UDA":
        algo = pg.algorithm(bat.my_BAT(gen = Iter))
  
    elif algorithm == "SCA_UDA":
        algo = pg.algorithm(sca.my_SCA(gen = Iter)) 
        
    else:
        return None
    
    evolved_pop = algo.evolve(pop)
    
    s = solution()
    #---------------------------------
    s.fevals = evolved_pop.problem.get_fevals()
    extra_info = algo.get_extra_info()
    li = list(extra_info.split("$"))    # li[0]: fevals       li[1]: executionTime       li[2]: convergence
    res = li[2][1:-1].split(', ')
    div_list = []
    if Export_diversity == True:
        div_list = li[3][1:-1].split(', ')
    # String to list   https://www.tutorialspoint.com/convert-a-string-representation-of-list-into-list-in-python
    res = ast.literal_eval(li[2])
    
    champion_individual = evolved_pop.champion_x
    champion_individual_fitness = evolved_pop.champion_f   #better
    s.bestIndividual = champion_individual
    s.best = champion_individual_fitness
    s.optimizer = algo.get_name()
    s.objfname = function_name
    s.executionTime = float(li[1])
    s.convergence = res
    s.diversity = div_list
    
    #--------------------------------
    return s


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_diversity = export_flags["Export_diversity"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for for the cinvergence
    CnvgHeader = []
    
    # CSV Header for for best solution
    SolHeader = [] 
    
    # CSV Header for for the diversity
    DivHeader = []

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))
    
    for l in range(0, Iterations):
        DivHeader.append("Iter" + str(l + 1))
        
    for l in range(0, 100):
        SolHeader.append("d" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            bestfit = [0] * NumOfRuns
            convergence = [0] * NumOfRuns
            diversity = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            fevals = [0] * NumOfRuns
           
            #-------------------objectivefunc details -------------------
            
            func_details = mf_details.getFunctionDetails(objectivefunc[j])
            #udp = Benchmarks.Math_Benchmarks(func_details = func_details)
            udp = pg.cec2014(prob_id = func_details[1], dim = func_details[4])
            #print(func_details)            
            #-------------------Construct a problem----------------------------------------
            prob = pg.problem(udp)
            #----------------------------------------------------------------------
            for k in range(0, NumOfRuns):
                
                #----------- # Construct the initial population of candidate solutions for the problem----------
                pop = pg.population(prob, size = PopulationSize)
                #------------------------------------------------------------------
                x = selector(optimizer[i], pop, Iterations, func_details[0],Export_diversity)
                convergence[k] = x.convergence
                diversity[k] = x.diversity
                optimizerName = x.optimizer
                objfname = x.objfname
                bestIndividual = x.bestIndividual

                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  # just one time to write the header of the CSV file
                            header = np.concatenate(
                                [["Optimizer", "objfname", "bestFit", "ExecutionTime", "fevals"], CnvgHeader, SolHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  # at least one experiment
                        executionTime[k] = x.executionTime
                        bestfit[k] = x.best
                        fevals[k] = x.fevals
                        #print(x.optimizer , " " , x.objfname, " ", x.convergence )
                        a = np.concatenate(
                            [[x.optimizer, x.objfname, x.best[0], x.executionTime, x.fevals], x.convergence, x.bestIndividual]
                        )
                        writer.writerow(a)
                    out.close()
                    
                    print(optimizerName, "," , objfname, " Run ", k ,  func_details[0], " completed")
                #print("time" , executionTime)
                

            #executionTime = list(map(float, executionTime))    # Thaer        
                
            if Export == True:
                ExportToFile = results_directory + "experiment.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = np.concatenate(
                            [["Optimizer", "objfname", "bestFit", "ExecutionTime", "fevals"],CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True
                    
                    #avgExecutionTime = float("%0.6f" % (sum(executionTime) / NumOfRuns))
                    avgExecutionTime = float(sum(executionTime) / NumOfRuns)
                    #avgbestfit = float("%0.6f" % (sum(bestfit) / NumOfRuns))
                    avgbestfit = float(sum(bestfit) / NumOfRuns)
                    #avgfevals = float("%0.6f" % (sum(fevals) / NumOfRuns))
                    avgfevals = float(sum(fevals) / NumOfRuns)
                    '''avgConvergence = np.around(
                        np.mean(convergence, axis=0, dtype=np.float64), decimals=5
                    ).tolist()'''
                    avgConvergence = np.mean(convergence, axis=0, dtype=np.float64).tolist()
                    a = np.concatenate(
                        [[optimizerName, objfname, avgbestfit, avgExecutionTime, avgfevals],avgConvergence]
                    )
                    writer.writerow(a)
                out.close()

            if Export_diversity == True:
                ExportToFile = results_directory + "experiment_diversity.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = np.concatenate(
                            [["Optimizer", "objfname", "bestFit", "ExecutionTime", "fevals"],DivHeader]
                        )
                        writer.writerow(header)
                        Flag = True
                    
                    #avgExecutionTime = float("%0.6f" % (sum(executionTime) / NumOfRuns))
                    avgExecutionTime = float(sum(executionTime) / NumOfRuns)
                    #avgbestfit = float("%0.6f" % (sum(bestfit) / NumOfRuns))
                    avgbestfit = float(sum(bestfit) / NumOfRuns)
                    #avgfevals = float("%0.6f" % (sum(fevals) / NumOfRuns))
                    avgfevals = float(sum(fevals) / NumOfRuns)
                    '''avgDiversity = np.around(
                        np.mean(diversity, axis=0, dtype=np.float64), decimals=5
                    ).tolist()'''
                    avgDiversity = np.mean(diversity, axis=0, dtype=np.float64).tolist()
                    a = np.concatenate(
                        [[optimizerName, objfname, avgbestfit, avgExecutionTime, avgfevals],avgDiversity]
                    )
                    writer.writerow(a)
                out.close()
            

    if Export_convergence == True:
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Export_boxplot == True:
        box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
