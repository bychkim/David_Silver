from Easy21 import Easy21
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from time import sleep
import pickle

rng = np.random

class MCControl:
    '''
    Monte Carlo Control
    '''
    N_0 = 100.0
    #N_0 = 1.0
    #N_0 = 10.0
    dict_sa_N = {}
    dict_sa_Q = {}
    dict_s_V ={}
    game = None
    num_game = 0.0
    acc_reward =0.0
    avg_reward =[]

    def __init__(self):
        rng.seed(123)
        pass

    def get_control(self,state):
        #epsilon-greedy optimal control
        #epsilon = self.N_0 /(self.N_0 + self.getN(state,0)+ self.getN(state,1))
        epsilon = 0.1
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
            if i % 100000 == 0:
                print 'show'
                self.showV()
                self.saveQ()

    def simulate_once(self):
        self.num_game += 1
        self.game  = Easy21()
        sas = []
        state = self.game.get_state()                # initial state
        action = self.get_control(state)             # control
        sas.append((state,action))
        #print state, action
        state, reward = self.game.step(state,action) # environment gives the next state and reward
        while self.game.is_finish == False:
            #print state, reward, self.game.is_finish
            action = self.get_control(state)         # agent choose the epsilon greedy control
            sas.append((state,action))
            #print state, action
            state, reward = self.game.step(state,action) # environment gives the next state and reward
            #print state, reward, self.game.is_finish
        #self.game.print_state()
        self.acc_reward += reward

        # accumulate this episode in data structure N and Q
        for i,(state,action) in enumerate(sas):
            #print i, state, action, reward
            self.updateN(state,action)
            #there is no discount weights.
            self.updateQ(state,action,reward)
        avg_reward = self.acc_reward /self.num_game
        self.avg_reward.append(avg_reward)
        print '# of games played:' +str(int(self.num_game))+' average reward:' + str(avg_reward)
        #print self.dict_sa_N
        #print self.dict_sa_Q

    def updateN(self,state,action):
        key = (state,action)
        if key in self.dict_sa_N:
            self.dict_sa_N[key] += 1
            #print key, self.dict_sa_N[key]
        else:
            self.dict_sa_N[key] = 1

    def getN(self,state,action):
        key = (state,action)
        if key in self.dict_sa_N:
            return self.dict_sa_N[key]
        else:
            return 0

    def updateQ(self,state,action,reward):
        alpha =1.0/float(self.getN(state,action))
        key = (state,action)
        if key in self.dict_sa_Q:
            self.dict_sa_Q[key] = self.dict_sa_Q[key] + alpha * ( reward - self.dict_sa_Q[key])
        else:
            self.dict_sa_Q[key] = alpha * (reward )

    def getQ(self,state,action):
        key = (state,action)
        if key in self.dict_sa_Q:
            return self.dict_sa_Q[key]
        else:
            return 0
    def saveQ(self):
        pickle.dump( self.dict_sa_Q, open( "results/MC_action_value_Q_func.p", "wb" ) )

    def buildV(self):
        X = np.zeros((10,21),dtype=np.float)
        Y = np.zeros((10,21),dtype=np.float)
        Z = np.zeros((10,21),dtype=np.float)
        for i in xrange(0,10):
            for j in xrange(0,21):
                state = (i+1,j+1) # index start from zero
                self.dict_s_V[state] = np.max( (self.getQ(state,0),self.getQ(state,1)) ) #bellman optimal condition
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
        fig.suptitle('Monte Carlo Control', fontsize=20)
        ax.set_xticks(xrange(1,11))
        ax.set_yticks(xrange(1,22))

        fig.savefig('results/mcc.png')

        fig2 = plt.figure(2)
        fig2.clf()
        plt.plot(self.avg_reward)

        fig2.savefig('results/avg_reward.png')


        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
        #plt.show()




mcc = MCControl()
#mcc.simulate_once()
mcc.simulate(3000000)
mcc.showV()



#game = Easy21()
#game.interactive()