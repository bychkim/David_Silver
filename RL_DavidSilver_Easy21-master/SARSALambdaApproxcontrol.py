from Easy21 import Easy21
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from time import sleep
rng = np.random

class SarsaControl:
    '''
    Monte Carlo Control
    '''
    N_0 = 100.0
    #N_0 = 1.0
    #N_0 = 10.0
    dict_sa_N = {}
    dict_sa_Q = {}
    dict_sa_E = {} #eligibility trace
    dict_s_V ={}

    #add for Q function
    dict_sa_Q_opt = {}
    list_Q_diff = []

    game = None
    num_game = 0.0
    acc_reward =0.0
    avg_reward =[]
    discount_rate = 1.0
    #td_lambda = 1.0
    __td_lambda = 0.0
    w = []
    num_features  = 36
    def __init__(self, td_lambda=0.5):
        rng.seed(1234)
        self.__td_lambda = td_lambda
        self.dict_sa_Q_opt = pickle.load(open("results/MC_action_value_Q_func.p","rb"))
        self.dict_sa_N = {}
        self.dict_sa_Q = {}
        self.list_Q_diff = []
        self.avg_reward = []
        self.w = np.random.random(self.num_features)
        pass

    def get_control(self,state):
        #epsilon-greedy optimal control
        epsilon = 0.05
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
            if i % 1000 == 0:
                print 'show'
                self.showQdiff()
                self.showV()

    def simulate_once(self):
        #debug
        #print self.w
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


    '''
    def simulate_once(self):
        self.num_game += 1
        self.game  = Easy21()
        sas = []

        state = self.game.get_state()
        action = self.get_control(state)
        next_state, reward = self.game.step(state,action)
        next_action = self.get_control(next_state)

        self.updateN(state,action)
        self.updateE(state,action)
        self.updateQ(state,action,reward,next_state,next_action)
        state  = next_state
        action = next_action

        #print state, reward, self.game.is_finish
        while self.game.is_finish == False:
            next_state, reward = self.game.step(state,action)
            next_action = self.get_control(next_state)
            self.updateN(state,action)
            self.updateE(state,action)
            self.updateQ(state,action,reward,next_state,next_action)
            state  = next_state
            action = next_action

        self.acc_reward += reward
        avg_reward = self.acc_reward /self.num_game
        self.avg_reward.append(avg_reward)
        print '# of games played:' +str(int(self.num_game))+' average reward:' + str(avg_reward)
    '''

    def updateE(self,state,action):
        key = (state,action)
        if key in self.dict_sa_E:
            self.dict_sa_E[key] += np.ones((self.num_features))
        else:
            self.dict_sa_E[key] = np.ones((self.num_features))

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

    def updateQ(self,state,action,reward,next_state,next_action):
      delta = reward + self.discount_rate * self.getQ(next_state,next_action) - self.getQ(state,action)
      alpha = 0.01
      #alpha = 0.001
      for i in range(1,11):
        for j in range(1,22):
          for a in range(0,2):
             s = (i,j)  # state
             if s == state and a == action:
                #print (s,a)
                self.updateSpecificE(s,a)
             else:
                self.degradeSpecificE(s,a)

             self.w = self.w + alpha * delta * self.getE(s,a)

    def updateSpecificE(self,state,action):
        key = (state, action)
        if key in self.dict_sa_E:
            #print self.dict_sa_E[key]
            self.dict_sa_E[key] = self.discount_rate * self.__td_lambda *self.dict_sa_E[key] + self.getFeature(state,action)
        else:
            self.dict_sa_E[key] = self.getFeature(state,action)

    def degradeSpecificE(self,state,action):
        key = (state, action)
        if key in self.dict_sa_E:
            self.dict_sa_E[key] = self.discount_rate * self.__td_lambda *self.dict_sa_E[key]
        else:
            self.dict_sa_E[key] = 0

    def in_range(self, bounds, x):
        # bounds is a tuple in ascending order(ex:(1,4))
        # from yoonho lee's code
        return x >= bounds[0] and x <= bounds[1]

    '''
    # the right feature function from yoonho lee's code
    def getFeature(self,state,action):
        # from yoonho lee's code
        features = []
        dealer_bounds = [(1,4), (4,7), (7,10)]
        player_bounds = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]
        actions = [0, 1]
        feature_bounds = list(itertools.product(dealer_bounds,player_bounds))
        feature_bounds = list(itertools.product(feature_bounds,actions))
        for feature_bound in feature_bounds:
            # feature_bound example: (((1,4), (13,18)), 'stick')
            feature_bool = self.in_range(feature_bound[0][0], state[0])\
                            and self.in_range(feature_bound[0][1], state[1])\
                            and action is feature_bound[1]
            features.append(int(feature_bool))

        return np.asarray(features,dtype=np.float)
    '''


    def getFeature(self,state,action):
        (dealer, player) = state
        d = -1
        p = -1
        a = -1
        if dealer >= 0 and dealer <=4:
            d = 0
        elif dealer >= 4 and dealer <=7:
            d = 1
        elif dealer >= 7 and dealer <=10:
            d = 2

        if player >=1 and player <= 6:
            p = 0
        elif player >=4 and player <= 9:
            p = 1
        elif player >=7 and player <= 12:
            p = 2
        elif player >=10 and player <= 15:
            p = 3
        elif player >=13 and player <= 18:
            p = 4
        elif player >=16 and player <= 21:
            p = 5

        if action == 1:
            a = 0
        elif action == 0:
            a = 1

        indication = np.zeros((36),dtype= float)

        if d == -1 or p == -1 or a ==-1:
            return indication

        #print np.shape(indication)
        #print index
        index = (d * 6 *2) + (p * 2) + a

        #print index

        indication[index] = 1.0
        return indication

    def getQ(self,state,action):
        feature = self.getFeature(state,action)
        #print feature
        #print np.shape(feature)
        #print np.shape(self.w)
        return np.dot(self.w,feature)


    def calQdiff(self):
        diff = 0.0
        for key in self.dict_sa_Q_opt.keys():
            diff += ( self.getQ(key[0],key[1]) - self.dict_sa_Q_opt[key])**2
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
                print (self.getQ(state,0),self.getQ(state,1))
                self.dict_s_V[state] = np.max( (self.getQ(state,0),self.getQ(state,1)) )
                X[i,j] = i+1
                Y[i,j] = j+1
                Z[i,j] = self.dict_s_V[state]
        return X,Y,Z


    def showV(self):
        X,Y,Z = self.buildV()
        print self.w
        #print self.dict_s_V
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

        fig.savefig('results/slc.png')

        fig2 = plt.figure(2)
        fig2.clf()
        plt.plot(self.avg_reward)

        fig2.savefig('results/slc_avg_reward.png')


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
fig4.savefig('results/approx_q_diff_per_lambda.png')

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

fig5.savefig('results/approx_list_Q_diff_zero_vs_one.png')
