from pebl import data, result
from pebl.learner import greedy, simanneal
import numpy as np
from pebl.taskcontroller import multiprocess
dataset = data.fromfile("pebl-tutorial-data1.txt")
# dt = np.loadtxt("/Users/jamespino/STOCH/pysb.examples.earm_1_3_16246_runfile42.gdat",dtype=str,skiprows=0)
#
# # dt[0,:] = dt[0,:].astype('string')
# # for i in range(len(dt[0,:])):
# #     dt[0, i] = 'S'+str(i)
# dt = dt[:,-3:]
# dt[0,0]='tBid_total'
# dt[0,1]='CPARP_total'
# dt[0,2]='cSmac_total'
#
#
# np.savetxt('test.csv',dt,delimiter='\t',fmt='%s')
#
#
# dataset = data.fromfile("test.csv")

dataset.discretize()
print('done')
learner1 = greedy.GreedyLearner(dataset, max_iterations=100)

print('done')
result1 = learner1.run()

learner2 = greedy.GreedyLearner(dataset, max_iterations=100)

result2 = learner2.run()

#learners = [ greedy.GreedyLearner(dataset, max_iterations=100) for i in range(5) ] #+ \
           # [ simanneal.SimulatedAnnealingLearner(dataset) for i in range(5) ]


#tc = multiprocess.MultiProcessController(poolsize=2)
#results = tc.run(learners)
#print(results)
#results = tc.run([learner1, learner2])

#results = tc.run(learners)
#print(results)
#merged_result = result.merge(results)
merged_result = result.merge(result1, result2)
merged_result.tohtml("example_merged")
merged_result.tofile('example.out')