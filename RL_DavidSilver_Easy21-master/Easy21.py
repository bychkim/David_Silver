'''
Implemented by Suwon Suh for Posco ZRM project
'''
import numpy as np
rng = np.random

class Easy21:
    """ Simplified Blackjack Game Class  """
    num_play = 0
    num_step = 1
    state_dealer = 0
    state_player = 0
    is_finish    = False
    reward = 0

    def draw_card(self):
        number = rng.randint(10)+1
        if rng.random() > (1.0/3.0):
            rb = 1  #-1:red, 1:black
        else:
            rb = -1
        return number,rb

    def __init__(self):
        self.num_play += 1
        number, rb = self.draw_card()
        self.state_dealer += number #only black
        number, rb = self.draw_card()
        self.state_player += number #only black

    def print_state(self):
        print 'step:' +str(self.num_step)
        print 'dealer state:' + str(self.state_dealer)
        print 'player state:' + str(self.state_player)
        print 'reward:' + str(self.reward)
        if self.is_finish:
            print 'Finished!'
        else:
            print 'On game.'
    def get_state(self):
        #self.print_state()
        return (self.state_dealer,self.state_player)

    def step(self,state , action):
        #self.print_states()
        self.num_step += 1
        if action == 0:
            self.reward = self.finalize()
            return (self.state_dealer,self.state_player), self.reward
        elif action == 1:
            number, rb = self.draw_card()
            self.state_player += number * rb
            if self.state_player > 21:
                self.reward = -1
                self.is_finish = True
            if self.state_player < 1:
                self.reward = -1
                self.is_finish = True
            return (self.state_dealer,self.state_player), self.reward
    def finalize(self):
        self.is_finish = True
        while self.state_dealer < 17:
            number, rb = self.draw_card()
            self.state_dealer += number * rb
            if self.state_dealer < 1:
                return 1
        if self.state_dealer > 21:
            return 1
        if self.state_dealer < self.state_player:
            return 1
        elif self.state_dealer == self.state_player:
            return 0
        elif self.state_dealer > self.state_player:
            return -1
    def interactive(self):
        while self.is_finish == False:
            state = self.get_state()
            action = input("stick or hit(0/1):")
            state, reward = self.step(state, action)
        self.get_state()

#main
#game = Easy21()
#game.interactive()
#print game.is_finish

#state = game.get_state()
#action = 1 #0:stick, 1:hit
#state, reward = game.step(state, action=0)
#state = game.get_state()


#easy21.step(1,1)
#print easy21.draw_card()
