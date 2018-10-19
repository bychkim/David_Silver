from Easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from time import sleep
import pickle

rng = np.random


'''
I have to rewrite this code to handle private variable later using variable mangling(?) using naming convention with prefix __.
'''


class SarsaControl:
    '''
    Monte Carlo Control
    '''
    N_0 = 100.0
    #N_0 = 1.0
    #N_0 = 10.0
    dict_sa_N = {}
    dict_sa_Q = {}

    #add for Q function
    dict_sa_Q_opt = {}
    list_Q_diff = []

    dict_sa_E = {} #eligibility trace
    dict_s_V ={}
    game = None
    num_game = 0.0
    acc_reward =0.0
    avg_reward =[]
    discount_rate = 1.0
    __td_lambda = 0.0


    def __init__(self, td_lambda=0.5):
        rng.seed(123)
        self.__td_lambda = td_lambda
        self.dict_sa_Q_opt = pickle.load(open("results/MC_action_value_Q_func.p","rb"))
        self.dict_sa_N = {}
        self.dict_sa_Q = {}
        self.list_Q_diff = []
        self.avg_reward = []

        pass

    def get_control(self,state):
        #epsilon-greedy optimal control
        epsilon = self.N_0 /(self.N_0 + self.getN(state,0)+ self.getN(state,1))
        #epsilon = 0.1
        if( np.random.random() > float(epsilon)):
            #optimal policy
            if(self.getQ(state,0) > self.getQ(state,1)):
                return 0
            else:
                return 1
        else:
            #random policy
            if(np.random.random() > 0.5):
                return 0
            else:
                return 1


    def simulate(self, num_iters):
        for i in xrange(num_iters):
            self.simulate_once()
            self.calQdiff()
            if i % 10000 == 0:
                print 'show'
                self.showQdiff()
                self.showV()

    def simulate_once(self):
        # game statistics
        self.num_game += 1
        # init eligibility trace
        self.dict_sa_E = {}
        # new game
        self.game = Easy21()
        state = self.game.get_state()
        action = self.get_control(state)
        while self.game.is_finish == False:
            self.updateN(state,action)
            self.updateE(state,action)
            next_state, reward = self.game.step(state,action) # environment
            next_action = self.get_control(next_state)        # e-greedy policy
            self.updateQ(state,action,reward,next_state,next_action)
            state = next_state
            action = next_action

        #debug
        self.acc_reward += reward
        avg_reward = self.acc_reward /self.num_game
        self.avg_reward.append(avg_reward)
        print '# of games played:' +str(int(self.num_game))+' average reward:' + str(avg_reward)

    def updateE(self,state,action):
        key = (state,action)
        if key in self.dict_sa_E:
            self.dict_sa_E[key] += 1
        else:
            self.dict_sa_E[key] = 1
    def getE(self,state,action):
        key = (state,action)
        if key in self.dict_sa_E:
            return self.dict_sa_E[key]
        else:
            return 0

    def updateN(self,state,action):
        key = (state,action)
        if key in self.dict_sa_N:
            self.dict_sa_N[key] += 1
        else:
            self.dict_sa_N[key] = 1

    def getN(self,state,action):
        key = (state,action)
        if key in self.dict_sa_N:
            return self.dict_sa_N[key]
        else:
            return 0

    '''
    def updateQ(self,state,action,reward,next_state,next_action):
      alpha =1.0/float(self.getN(state,action))
      key = (state,action)
      if next_state == None:
        if key in self.dict_sa_Q:
            self.dict_sa_Q[key] = self.dict_sa_Q[key] + alpha * ( reward - self.dict_sa_Q[key])
        else:
            self.dict_sa_Q[key] = alpha * (reward )
      else:
        next_key = (next_state,next_action)
        if next_key not in self.dict_sa_Q:
            self.dict_sa_Q[next_key] = 0 # initialize next state if it doesn't exist
        if key in self.dict_sa_Q:
            self.dict_sa_Q[key] = self.dict_sa_Q[key] + alpha * ( reward +(self.discount_rate * self.dict_sa_Q[next_key])- self.dict_sa_Q[key])
        else:
            self.dict_sa_Q[key] = alpha * (reward )
    '''

    def updateQ(self,state,action,reward,next_state,next_action):
      delta = reward + self.discount_rate * self.getQ(next_state,next_action) - self.getQ(state,action)
      for i in range(1,11):
        for j in range(1,22):
          for a in range(0,2):
             s = (i,j)
             alpha =1.0/float(1.0 + self.getN(s,a))
             #self.updateSpecificQ(s,a,alpha*delta)
             self.updateSpecificQ(s,a,alpha*delta*self.getE(s,a))
             self.updateSpecificE(s,a)


    def updateSpecificQ(self,state,action,value):
        key = (state,action)
        if key in self.dict_sa_Q:
            self.dict_sa_Q[key] += value
        else:
            self.dict_sa_Q[key] = value

    def updateSpecificE(self,state,action):
        key = (state, action)
        if key in self.dict_sa_E:
            self.dict_sa_E[key] = self.discount_rate * self.__td_lambda *self.dict_sa_E[key]
        else:
            self.dict_sa_E[key] = 0

    def getQ(self,state,action):
        key = (state,action)
        if key in self.dict_sa_Q:
            return self.dict_sa_Q[key]
        else:
            return 0

    def calQdiff(self):
        diff = 0.0
        for key in self.dict_sa_Q_opt.keys():
            if key in self.dict_sa_Q:
                diff += ( self.dict_sa_Q[key] - self.dict_sa_Q_opt[key])**2
            else:
                diff += ( self.dict_sa_Q_opt[key])**2
        print "Q difference from optimal Q: %f"  % (diff)
        self.list_Q_diff.append(diff)

    def showQdiff(self):
        fig3 = plt.figure(3)
        fig3.clf()
        plt.plot(self.list_Q_diff)

        fig3.savefig('results/list_Q_diff.png')


    def buildV(self):
        X = np.zeros((10,21),dtype=np.float)
        Y = np.zeros((10,21),dtype=np.float)
        Z = np.zeros((10,21),dtype=np.float)
        for i in xrange(0,10):
            for j in xrange(0,21):
                state = (i+1,j+1) # index start from zero
                self.dict_s_V[state] = np.max( (self.getQ(state,0),self.getQ(state,1)) )
                X[i,j] = i+1
                Y[i,j] = j+1
                Z[i,j] = self.dict_s_V[state]
        return X,Y,Z

    def showV(self):
        X,Y,Z = self.buildV()
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        plt.xlim([1,10])
        plt.xlabel('dealer showing', fontsize=18)
        plt.ylim([1,21])
        plt.ylabel('player sum', fontsize=18)
        fig.suptitle('Sarsa(\lambda) Control', fontsize=20)
        ax.set_xticks(xrange(1,11))
        ax.set_yticks(xrange(1,22))

        fig.savefig('results/srlc.png')

        fig2 = plt.figure(2)
        fig2.clf()
        plt.plot(self.avg_reward)

        fig2.savefig('results/srlc_avg_reward.png')


        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
        #plt.show()

############## Experiment 1 ###########################
# 0.0 to 1.0
list_lambda_Q = []
for i in xrange(11):
    td_lambda = float(i)/10.0
    sc = SarsaControl(td_lambda)
    sc.simulate(10000)
    list_lambda_Q.append( sc.list_Q_diff[-1])
    del sc

fig4 = plt.figure(4)
plt.plot(list_lambda_Q)
plt.xticks([0,1,2,3,4,5,6,7,8,9.10], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.draw()
fig4.savefig('results/q_diff_per_lambda.png')

############## Experiment 2 ##########################
# 0.0 vs 1.0
sc = SarsaControl(td_lambda=1.0)
sc.simulate(10000)
list_Q_diff_one =  sc.list_Q_diff
del sc

sc = SarsaControl(td_lambda=0.0)
sc.simulate(10000)
list_Q_diff_zero =  sc.list_Q_diff
del sc

fig5 = plt.figure(5)
plt.plot(list_Q_diff_zero,label= 'zero')
plt.plot(list_Q_diff_one, label = 'one')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('sum of square error')
plt.title('comparison between zero vs one ')

plt.draw()

fig5.savefig('results/list_Q_diff_zero_vs_one.png')
