from pebl import data, result
from pebl.learner import greedy, simanneal
import numpy as np
from pebl.taskcontroller import multiprocess
dataset = data.fromfile("zinc.txt")


dataset.discretize()

learner1 = greedy.GreedyLearner(dataset, max_iterations=1000)

result1 = learner1.run()

learner2 = greedy.GreedyLearner(dataset, max_iterations=1000)

result2 = learner2.run()

#learners = [ greedy.GreedyLearner(dataset, max_iterations=100) for i in range(5) ] #+ \
           # [ simanneal.SimulatedAnnealingLearner(dataset) for i in range(5) ]

result1.tohtml('example2.out')
result2.tohtml('example3.out')
merged_result = result.merge(result1, result2)
merged_result.tohtml("zinc",)
merged_result.tofile('example.out')