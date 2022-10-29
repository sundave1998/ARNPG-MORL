import numpy as np


class MDP:
    def __init__(self, reward, utility, s, a, rho, gamma, prob_transition, model=None):
        
        self.reward = reward
        self.utility = utility
        self.s = s
        self.a = a
        self.rho = rho
        self.gamma = gamma
        self.prob_transition = prob_transition
        self.model = model
        
        self.theta = np.random.uniform(0, 1, size=s * a)
        self.gap = []
        self.acc_avg_gap = 0
        self.iter_num = 0



    def theta_to_policy(self):
        """
        :param theta: |S||A| * 1
        :param s: |S|
        :param a: |A|
        :return: |S||A| * 1
        """
        prob = []
        for i in range(self.s):
            norm = np.sum(np.exp(self.theta[self.a * i:self.a * (i + 1)]))
            for j in range(self.a * i, self.a * (i + 1)):
                prob.append(np.exp(self.theta[j]) / norm)

        return np.asarray(prob)


    def get_Pi(self, prob):
        """
        :param prob: |S||A| * 1
        :param s: |S|
        :param a: |A|
        :return: |S| * |S||A|
        """
        Pi = np.zeros((self.s, self.s * self.a))
        for i in range(self.s):
            Pi[i, i * self.a:(i + 1) * self.a] = prob[i * self.a:(i + 1) * self.a]

        return Pi


    def grad_state_action(self, prob, state, action):
        """
        :param prob: |S||A| * 1
        :param state: 1 * 1
        :param action: 1 * 1
        :return: \nabla_{\theta} \pi_{\theta}(s,a)
        """
        grad = np.zeros(self.s * self.a)
        for j in range(0, self.a):
            if j == action:
                grad[self.a * state + j] = prob[self.a * state + j] * (1 - prob[self.a * state + j])
            else:
                grad[self.a * state + j] = -prob[self.a * state + action] * prob[self.a * state + j]

        return grad


    def grad_state(self, qvals, prob, state):
        grad = np.sum([qvals[state * self.a + i] * self.grad_state_action(prob, state, i) for i in range(0, self.a)], axis=0)
        return grad


    def grad(self, qvals, prob, d_pi):
        grad = np.sum([d_pi[i] * self.grad_state(qvals, prob, i) for i in range(0, self.s)], axis=0)
        return grad


    def Fisher_info(self, prob, d_pi):
        """
        :param prob: |S||A| * 1
        :param d_pi: |S| * 1
        :return: Fisher information matrix \nabla_{\theta} \pi_{\theta}(s,a) x {\nabla_{\theta} \pi_{\theta}(s,a)}^T
        """
        qvals_one = np.ones(self.s * self.a)
        grad = np.sum([d_pi[i] * self.grad_state(qvals_one, prob, i) for i in range(0, self.s)], axis=0)
        fisher = np.outer(grad, grad) + 1e-3 * np.identity(self.s * self.a)
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
        for i in range(self.s):
            V[i] = np.sum([qvals[i * self.a + j] * prob[i * self.a + j] for j in range(self.a)])

        ell = np.dot(V, self.rho)
        return ell


#     def proj(self, scalar, offset = 100):
# #         offset = 100
#         if scalar < 0:
#             scalar = 0

#         if scalar > offset:
#             scalar = offset

#         return scalar

    def Q_cal(self, V, func):
        """
        Calculate Q from V value function
        :param V: |S| * 1
        :param func: reward/cost function |S||A| * 1
        :return: Q |S||A| * 1
        """
        Q = np.zeros(self.s * self.a)
        for i in range(self.s):
            for j in range(self.a):
                Q[i * self.a + j] = func[i * self.a + j] + self.gamma * np.matmul(self.prob_transition[i * self.a + j], V)
        return Q


class NPG_MDP(MDP):
    
    
    def __init__(self, reward, utility, s, a, rho, gamma, prob_transition, model=None):
        
            
        super().__init__(reward, utility, s, a, rho, gamma, prob_transition, model)
        self.div_number = 1
        self.step = 4.5
        self.V1 = []
        self.V2 = []

    def NPG_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.reward))
        Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))

        # V(\rho): 1*1
        vrvals = np.dot(np.transpose(Vr), self.rho)
        vgvals = np.dot(np.transpose(Vg), self.rho)
        V = Vr / vrvals + Vg / vgvals
        vvals = np.dot(np.transpose(V), self.rho)

        qrvals = self.Q_cal(Vr, self.reward)
        qgvals = self.Q_cal(Vg, self.utility)
        qvals = self.Q_cal(V, self.reward / vrvals + self.utility / vgvals)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0 and self.model is not None:

            avg_reward = self.ell(qrvals, prob)
            avg_utility = self.ell(qgvals, prob)

            self.acc_avg_gap += np.log(self.model.objective_value) - np.log(avg_reward*avg_utility)
#             self.model.objective_value - avg_reward
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.gap.append(self.acc_avg_gap / (self.iter_num))
            self.V1.append(avg_reward)
            self.V2.append(avg_utility)
            
            
class MO_MDP(MDP):
    
    def __init__(self, rewards, s, a, rho, gamma, prob_transition, model=None):
        self.rewards = rewards
#         self.utility = utility
        self.s = s
        self.a = a
        self.rho = rho
        self.gamma = gamma
        self.prob_transition = prob_transition
        self.model = model
        
        self.theta = np.random.uniform(0, 1, size=s * a)
        
        self.acc_avg_gap = 0
        self.iter_num = 0
            
#         super().__init__(reward, utility, s, a, rho, gamma, prob_transition, model)
        self.div_number = 1
        self.step = 4.5
        
        self.obj_num = len(rewards)
        self.Q_record = []
        self.V_record = [[] for _ in range(self.obj_num)]

    def Centralized_log_NPG_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        # Calculate for all objectives
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))

        # V(\rho): 1*1
        # Calculate for all objectives
#         v_vals_list = []
#         for i in range(self.obj_num):
#             v_vals_list.append(np.dot(np.transpose(V_list[i]), self.rho))
#         print(v_vals_list)    
            
        v_vals_list = np.zeros(self.obj_num)
        for i in range(self.obj_num):
            v_vals_list[i] = np.dot(np.transpose(V_list[i]), self.rho)
#         print(v_vals_list)    

        total_V = np.zeros(V_list[0].shape)
        for i in range(self.obj_num):
            total_V += V_list[i]/v_vals_list[i]
#         V = Vr / vrvals + Vg / vgvals
        vvals = np.dot(np.transpose(total_V), self.rho)

        q_vals_list = []
        for i in range(self.obj_num):
            q_vals_list.append(self.Q_cal(V_list[i], self.rewards[i]))
#         qrvals = self.Q_cal(Vr, self.reward)
#         qgvals = self.Q_cal(Vg, self.utility)
        global_reward = np.zeros(self.rewards[0].shape)
        for i in range(self.obj_num):
            global_reward += self.rewards[i]/v_vals_list[i]
        qvals = self.Q_cal(total_V, global_reward)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0 and self.model is not None:

            log_sum = 0
            for i in range(self.obj_num):
                self.V_record[i].append(self.ell(q_vals_list[i], prob))
                log_sum+=np.log(self.ell(q_vals_list[i], prob))
                
#             avg_reward = self.ell(qvals, prob)
#             avg_utility = self.ell(qgvals, prob)

            self.acc_avg_gap +=  log_sum
#             self.model.objective_value - avg_reward
            if verbose:
                print('Average:', self.acc_avg_gap / (self.iter_num))
            self.Q_record.append(self.acc_avg_gap / (self.iter_num))
            
#             for i in range(self.obj_num):
            
#             self.V1.append(avg_reward)
#             self.V2.append(avg_utility)
    def calculate_G(self, num=None, verbose=False):
        
#         self.iter_num += 1
        if verbose: print("Calculate G of iteration:", self.iter_num+1)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        # Calculate for all objectives
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))

        # V(\rho): 1*1
        # Calculate for all objectives
#         v_vals_list = []
#         for i in range(self.obj_num):
#             v_vals_list.append(np.dot(np.transpose(V_list[i]), self.rho))
#         print(num)
        v_vals_list = np.zeros(self.obj_num)
        if num==None:
            if verbose:
                print("return global gradient")
            for i in range(self.obj_num):
                v_vals_list[i] = 1/np.dot(np.transpose(V_list[i]), self.rho)
        else:
            if verbose:
                print("return local gradient")
            v_vals_list[num] = 1/np.dot(np.transpose(V_list[num]), self.rho)
        
        return v_vals_list
        

    def NPG_step_given_G(self, direction, verbose=False):
        
        self.iter_num += 1
        if verbose: print("step towards direction at iteration:", self.iter_num)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        # Calculate for all objectives
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))

        # V(\rho): 1*1
        # Calculate for all objectives
#         v_vals_list = []
#         for i in range(self.obj_num):
#             v_vals_list.append(np.dot(np.transpose(V_list[i]), self.rho))
#         print(v_vals_list)    
            
#         v_vals_list = np.zeros(self.obj_num)
#         for i in range(self.obj_num):
#             v_vals_list[i] = np.dot(np.transpose(V_list[i]), self.rho)
# #         print(v_vals_list)    

        total_V = np.zeros(V_list[0].shape)
        for i in range(self.obj_num):
            total_V += V_list[i]*direction[i]
#         V = Vr / vrvals + Vg / vgvals
        vvals = np.dot(np.transpose(total_V), self.rho)

        q_vals_list = []
        for i in range(self.obj_num):
            q_vals_list.append(self.Q_cal(V_list[i], self.rewards[i]))
#         qrvals = self.Q_cal(Vr, self.reward)
#         qgvals = self.Q_cal(Vg, self.utility)
        global_reward = np.zeros(self.rewards[0].shape)
        for i in range(self.obj_num):
            global_reward += self.rewards[i]*direction[i]
        qvals = self.Q_cal(total_V, global_reward)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0 and self.model is not None:

            log_sum = 0
            for i in range(self.obj_num):
                self.V_record[i].append(self.ell(q_vals_list[i], prob))
                log_sum+=np.log(self.ell(q_vals_list[i], prob))
                
#             avg_reward = self.ell(qvals, prob)
#             avg_utility = self.ell(qgvals, prob)

            self.acc_avg_gap +=  log_sum
#             self.model.objective_value - avg_reward
            if verbose:
                print('Average:', self.acc_avg_gap / (self.iter_num))
            self.Q_record.append(self.acc_avg_gap / (self.iter_num))
            
#             for i in range(self.obj_num):
            
#             self.V1.append(avg_reward)
#             self.V2.append(avg_utility)