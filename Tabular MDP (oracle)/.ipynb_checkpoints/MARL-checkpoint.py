import numpy as np


class MARL_agent:
    def __init__(self, rewards, s, a, rho, gamma, n, prob_transition, model=None):
        
        self.rewards = rewards
        self.s = s
        self.a = a
        self.n = n
        # new parameter global action space A
        self.A = a**n

        self.rho = rho
        self.gamma = gamma
        self.prob_transition = prob_transition
        self.model = model

        if self.prob_transition.shape[0] != self.A * s:
            print("dimension mismatch, global action space should be ", self.A)
        
        self.obj_num = len(rewards)
        self.theta = np.random.uniform(0, 1, size=n* s* a)
        # self.theta = np.random.uniform(0, 1, size=(n, s, a))
        
        self.gap = []
        self.acc_avg_gap = 0
        self.iter_num = 0
        self.step = 0.1
        self.div_number = 1


    def theta_to_policy(self):
        """
        :param theta: n*|S|*|a| 
        :param s: |S|
        :param a: |a|
        :param n: n total agent num
        :return: n*|S|*|a| 
        """
        # prob = np.zeros((self.n, self.s, self.a))
        # for k in range(self.n):
        #     for i in range(self.s):
        #         norm = np.sum(np.exp(self.theta[k,i,:]))
        #         for j in range(self.a):
        #             prob[k,i,j] = (np.exp(self.theta[k,i,j]) / norm)

        prob = np.zeros((self.n* self.s* self.a))
        for k in range(self.n):
            for i in range(self.s):
                norm = np.sum(np.exp(self.theta[k*(self.s*self.a)+i*self.a:k*(self.s*self.a)+(i+1)*self.a]))
                for j in range(self.a):
                    prob[k*(self.s*self.a)+i*self.a+j] = (np.exp(self.theta[k*(self.s*self.a)+i*self.a+j]) / norm)

        return prob


    def get_Pi(self, prob):
        """
        :param prob: n*|S|*|a| 
        :param s: |S|
        :param a: |a|
        :param n: n total agent num
        :return: |S| * |S||A|, A = a^n
        """
        # Pi = np.zeros((self.s, self.a**self.n))

        # for state in range(self.s):
        #     for action in range(self.a**self.n):
        #         # ind=np.zeros(n)
        #         temp = 1
        #         variable = action
        #         for i in range(self.n):
        #             # ind[i] = int(variable%a)
                    
        # #             print(i,state,ind[i])
        #             temp *= prob[i,state,int(variable%self.a)]
        #             variable = int(variable/self.a)
        # #         print(ind)
        #         Pi[state,action] = temp

        Pi = np.zeros((self.s, self.s*self.A))

        for state in range(self.s):
            for action in range(self.A):
                # ind=np.zeros(n)
                temp = 1
                variable = action
                for i in range(self.n):
                    # ind[i] = int(variable%a)
                    
        #             print(i,state,ind[i])
                    temp *= prob[i*self.s*self.a + state*self.a + int(variable%self.a)]
                    variable = int(variable/self.a)
        #         print(ind)
                Pi[state,(state*self.A)+action] = temp
        return Pi


        # for i in range(self.s):
            
        #     Pi[i, i * self.a:(i + 1) * self.a] = prob[i * self.a:(i + 1) * self.a]

        # return Pi


    def grad_state_action(self, prob, state, action):
        """
        :param prob: n|S||a| * 1
        :param state: 1 * 1
        :param action: 1 * 1
        :return: nabla_{theta} \pi_{theta}(s,A)
        """
        grad = np.zeros(self.n* self.s * self.a)
        variable = action
        local_action=np.zeros(self.n, dtype=np.int8)

        for i in range(self.n):
            local_action[i] = int(variable%self.a)
            variable = int(variable/self.a)
        for i in range(self.n):
            for j in range(self.a):
                multiplier = 1
                for k in range(self.n):
                    if k!=i:
                        multiplier = multiplier*prob[int(k*self.a*self.s +self.a * state + local_action[k])]
                if j == local_action[i]:
                    grad[i*self.a*self.s + self.a * state + j] = multiplier * prob[i*self.a*self.s + self.a * state + j] * (1 - prob[i*self.a*self.s + self.a * state + j])
                else:
                    grad[i*self.a*self.s + self.a * state + j] = -multiplier * prob[i*self.a*self.s + self.a * state + local_action[i]] * prob[i*self.a*self.s + self.a * state + j]

        return grad


    def grad_state(self, qvals, prob, state):
        grad = np.sum([qvals[state * self.A + i] * self.grad_state_action(prob, state, i) for i in range(0, self.A)], axis=0)
        return grad


    def grad(self, qvals, prob, d_pi):
        grad = np.sum([d_pi[i] * self.grad_state(qvals, prob, i) for i in range(0, self.s)], axis=0)
        return grad


    def Fisher_info(self, prob, d_pi):
        """
        :param prob: |S||A| * 1
        :param d_pi: |S| * 1
        :return: Fisher information matrix nabla_{theta} pi_{theta}(s,a) x {nabla_{theta} \pi_{theta}(s,a)}^T
        """
        qvals_one = np.ones(self.s * self.A)
        grad = np.sum([d_pi[i] * self.grad_state(qvals_one, prob, i) for i in range(0, self.s)], axis=0)
        # grad: n*s*a

        fisher = np.outer(grad, grad) + 1e-3 * np.identity(self.n* self.s * self.a)
        return fisher


    def ell(self, qvals, prob):
        """
        Calculate V from Q value function
        :param qvals: |S||A| * 1
        :param prob: |S||A| * 1
        :param rho: |S| * 1
        :return: V |S| * 1
        """
        V = np.zeros(self.s)
        for state in range(self.s):
            for action in range(self.A):
                temp = 1
                variable = action
                for i in range(self.n):
                    # ind[i] = int(variable%a)
                    
        #             print(i,state,ind[i])
                    temp *= prob[i*self.s*self.a + state*self.a + int(variable%self.a)]
                    variable = int(variable/self.a)
        #         print(ind)
                # Pi[state,(state*self.A)+action] = 
                V[state]+=temp*qvals[state * self.A + action]
            # V[i] = np.sum([qvals[i * self.A + j] * prob[i * self.A + j] for j in range(self.A)])

        ell = np.dot(V, self.rho)
        return ell


    # def proj(self, scalar, offset = 100):
    #     if scalar < 0:
    #         scalar = 0

    #     if scalar > offset:
    #         scalar = offset

    #     return scalar

    def Q_cal(self, V, func):
        """
        Calculate Q from V value function
        :param V: |S| * 1
        :param func: reward/cost function |S||A| * 1
        :return: Q |S||A| * 1
        """
        Q = np.zeros(self.s * self.A)
        for i in range(self.s):
            for j in range(self.A):
                Q[i * self.A + j] = func[i * self.A + j] + self.gamma * np.matmul(self.prob_transition[i * self.A + j], V)
        return Q


    def NPG_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0]))
        # Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))

        # V(\rho): 1*1
        vrvals = np.dot(np.transpose(Vr), self.rho)
        # vgvals = np.dot(np.transpose(Vg), self.rho)
        V = Vr 
        vvals = np.dot(np.transpose(V), self.rho)

        qrvals = self.Q_cal(Vr, self.rewards[0])
        # qgvals = self.Q_cal(Vg, self.utility)
        # qvals = self.Q_cal(V, self.reward / vrvals + self.utility / vgvals)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qrvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0:
            # print(qrvals)

            avg_reward = self.ell(qrvals, prob)
            # avg_utility = self.ell(qgvals, prob)

            self.acc_avg_gap +=  np.log(avg_reward)
            # self.model.objective_value - avg_reward
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.gap.append(self.acc_avg_gap / (self.iter_num))
            # self.V1.append(avg_reward)
            # self.V2.append(avg_utility)
            

