import numpy as np


class MARL_agent:
    def __init__(self, rewards, s, a, rho, gamma, n, prob_transition, model=None, tau=1):
        
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
        
        self.avg_gap = []
        self.gap = []
        self.acc_avg_gap = 0
        self.iter_num = 0
        self.tau = tau
        self.step = 0.01
        self.div_number = 1
        self.Q_tilde = []
        self.Q_global = []

        
    def theta_to_policy(self):
        """
        :param theta: n*|S|*|a| 
        :param s: |S|
        :param a: |a|
        :param n: n total agent num
        :return: n*|S|*|a| 
        """
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
        Pi = np.zeros((self.s, self.s*self.A))

        for state in range(self.s):
            for action in range(self.A):
                temp = 1
                variable = action
                for i in range(self.n):
                    temp *= prob[i*self.s*self.a + state*self.a + int(variable%self.a)]
                    variable = int(variable/self.a)
                Pi[state,(state*self.A)+action] = temp
        return Pi

        
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
                    temp *= prob[i*self.s*self.a + state*self.a + int(variable%self.a)]
                    variable = int(variable/self.a)
                V[state]+=temp*qvals[state * self.A + action]
        ell = np.dot(V, self.rho)
        return ell


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

    
    def tilde_cal(self, Q, prob, i):
        """
        Calculate Q_tilde from Q value function
        :param Q: |S||A| * 1
        :param prob: n*|S|*|a| 
        :param i: agent number
        :return: Q_tilde |S||a| * 1
        """
        Q_tilde = np.zeros(self.s * self.a)
        for state in range(self.s):
            for global_action in range(self.A):
                multiplier = 1
                temp = global_action
                for agent in range(self.n):
                    
                    local_action = temp%self.a
                    temp = int(temp/self.a)
                    if agent!=i:
                        multiplier*= prob[agent*(self.s*self.a)+state*self.a+local_action]
                    else:
                        agent_action = local_action
                Q_tilde[state*self.a+agent_action] += Q[state*self.A+global_action]*multiplier
        return Q_tilde

    
    def A_tau_cal(self, Q_tau, prob, V_tau, agent):
        """
        Calculate A from Q value function
        :param Q: |S||A| * 1
        :param V: |S| * 1
        :param prob: n*|S|*|a| 
        :param i: agent number
        :return: A |S||A| * 1
        """
        A_tau = np.copy(Q_tau)
        for state in range(self.s):
            for action in range(self.A):
                if agent == -1:
                    temp = action
                    multiplier = 1
                    for i in range(self.n):
                        local_action = temp%self.a
                        temp = int(temp/self.a)
                        A_tau[state*self.A+action] -= self.tau*np.log(prob[i*(self.s*self.a)+state*self.a+local_action])

                    A_tau[state*self.A+action] -= V_tau[state]
                else:
                    temp = action
                    for i in range(self.n):
                        local_action = temp%self.a
                        temp = int(temp/self.a)
                        if agent==i:
                            agent_action = local_action
                            break
                    A_tau[state*self.A+action] -= (self.tau*np.log(prob[agent*(self.s*self.a)+state*self.a+local_action]) + V_tau[state])
        return A_tau


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

        # V(\rho): 1*1
        vrvals = np.dot(np.transpose(Vr), self.rho)
        V = Vr 
        vvals = np.dot(np.transpose(V), self.rho)

        qrvals = self.Q_cal(Vr, self.rewards[0])

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qrvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0:
            avg_reward = self.ell(qrvals, prob)
            self.acc_avg_gap +=  np.log(avg_reward)
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.avg_gap.append(self.acc_avg_gap / (self.iter_num))
            self.gap.append(np.log(avg_reward))


    def NPG_entropy_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition) # |S|*|S|
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        d_s = np.linalg.inv(np.identity(self.s) - self.gamma * P_theta)
        Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0]))
        
        V_taus = []
        V_tau_global = np.zeros(self.s)
        log_prob = np.zeros(self.n * self.s * self.A)
        for state in range(self.s):
            for action in range(self.A):
                temp = action
                for agent in range(self.n):
                    local_action = temp%self.a
                    temp = int(temp/self.a)
                    log_prob[agent*self.s*self.A+state*self.A+action] = np.log(prob[agent*self.s*self.a+state*self.a+local_action])    
        
        for agent in range(self.n):    
            V_tau_i = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0] - self.tau * log_prob[agent]))
            V_taus.append(V_tau_i)
            V_tau_global += V_tau_i

        # V(\rho): 1*1
        vrvals = np.dot(np.transpose(Vr), self.rho)
        V = Vr 
        vvals = np.dot(np.transpose(V), self.rho)
        v_tau_vals = np.dot(np.transpose(V_tau_global), self.rho)

        qrvals = self.Q_cal(Vr, self.rewards[0] * self.n)
        q_tau_global = self.Q_cal(V_tau_global, self.rewards[0] * self.n)
        q_taus = []
        Q_tildes = []
        A_taus = []
        A_tildes = []
        for agent in range(self.n):
            q_tau_i = self.Q_cal(V_taus[agent], self.rewards[0])
            q_taus.append(q_tau_i)
            Q_tildes.append(self.tilde_cal(q_tau_i, prob, agent))
            A_tau = self.A_tau_cal(q_tau_i, prob, V_taus[agent], agent)
            A_taus.append(A_tau)
            A_tildes.append(self.tilde_cal(A_tau, prob, agent))
#         print('test', np.dot(A_tildes[0], prob[0:self.s*self.a]))
    
        for agent in range(self.n):
            self.theta[agent*(self.s*self.a):(agent+1)*(self.s*self.a)] += self.step * (1/(1-self.gamma))* A_tildes[agent]

        if verbose:
            print("A_tildes", A_tildes)
            print("theta", self.theta)
            print("prob", prob[:])
            
        if self.iter_num % self.div_number == 0:
            avg_reward = v_tau_vals
            self.acc_avg_gap += (avg_reward)

            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.avg_gap.append(self.acc_avg_gap / (self.iter_num))
            self.gap.append((avg_reward))
      
        
    def NPG_entropy_step_change(self, verbose=False):
        
        def cal_Q_tildes(Atildes, Pi, Vtau, tau=0.1):
            Q_tildes = np.zeros(self.n*self.s*self.a)
            for agent in range(self.n):
                for state in range(self.s):
                    Q_tildes[agent*self.s*self.a+state*self.a:agent*self.s*self.a+(state+1)*self.a] = Atildes[agent][state*self.a:(state+1)*self.a] + tau * np.log(Pi[state*self.a:(state+1)*self.a]) + Vtau[state]
            return Q_tildes
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition) # |S|*|S|
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        d_s = np.linalg.inv(np.identity(self.s) - self.gamma * P_theta)
        Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0]))
        
        log_prob = np.zeros(self.s * self.A)
        for state in range(self.s):
            for action in range(self.A):
                temp = action
                for agent in range(self.n):
                    local_action = temp%self.a
                    temp = int(temp/self.a)
                    log_prob[state*self.A+action] += np.log(prob[agent*self.s*self.a+state*self.a+local_action])         
        V_tau = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0] - self.tau * log_prob))

        # V(\rho): 1*1
        vrvals = np.dot(np.transpose(Vr), self.rho)
        v_tau_vals = np.dot(np.transpose(V_tau), self.rho)
        q_tau = self.Q_cal(V_tau, self.rewards[0])
        A_tildes = []        
        A_tau = self.A_tau_cal(q_tau, prob, V_tau, -1)
        
        for agent in range(self.n):
            A_tildes.append(self.tilde_cal(A_tau, prob, agent))
        self.Q_tilde.append(cal_Q_tildes(A_tildes, prob, V_tau, tau=self.tau))
        self.Q_global.append(q_tau)

        for agent in range(self.n):
            self.theta[agent*(self.s*self.a):(agent+1)*(self.s*self.a)] += self.step * (1/(1-self.gamma))* A_tildes[agent]

        if verbose:
            print("A_tildes", A_tildes)
            print("theta", self.theta)
            print("prob", prob[:])

        if self.iter_num % self.div_number == 0:
            avg_reward = v_tau_vals
            self.acc_avg_gap +=  (avg_reward)
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.avg_gap.append(self.acc_avg_gap / (self.iter_num))
            self.gap.append((avg_reward))

        return A_tildes, prob, V_tau, A_tau
    
    
    def solver(self, typ='global'):
        self.step = 0.1 * (1 - self.gamma) / self.tau
        for _ in range(10000):
            if typ == 'global':
                A_tildes, prob, V_tau, A_tau = self.NPG_entropy_step_change(verbose=False)
            else:
                self.NPG_entropy_step(verbose=False)
        if typ == 'global':
            return self.gap[-1], A_tildes, prob, V_tau
        else:
            return self.gap[-1]


class MO_MARL(MARL_agent):
    def __init__(self, rewards, s, a, rho, gamma, n, prob_transition, model=None):
        super().__init__(rewards, s, a, rho, gamma, n, prob_transition, model)
        self.V_record = [[] for _ in range(self.obj_num)]
        


    def Centralized_NPG_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))
        # Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0]))
        # Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))

        # V(\rho): 1*1
        v_vals_list = np.zeros(self.obj_num)
        for i in range(self.obj_num):
            v_vals_list[i] = np.dot(np.transpose(V_list[i]), self.rho)

        total_V = np.zeros(V_list[0].shape)
        for i in range(self.obj_num):
            total_V += V_list[i]/v_vals_list[i]
        vvals = np.dot(np.transpose(total_V), self.rho)
        # vrvals = np.dot(np.transpose(Vr), self.rho)
        # vgvals = np.dot(np.transpose(Vg), self.rho)
        # V = Vr 
        # vvals = np.dot(np.transpose(V), self.rho)
        q_vals_list = []
        for i in range(self.obj_num):
            q_vals_list.append(self.Q_cal(V_list[i], self.rewards[i]))
        # qrvals = self.Q_cal(Vr, self.rewards[0])
        # qgvals = self.Q_cal(Vg, self.utility)
        # qvals = self.Q_cal(V, self.reward / vrvals + self.utility / vgvals)

        global_reward = np.zeros(self.rewards[0].shape)
        for i in range(self.obj_num):
            global_reward += self.rewards[i]/v_vals_list[i]
        qvals = self.Q_cal(total_V, global_reward)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals-vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0:
            # print(qrvals)

            log_sum = 0
            for i in range(self.obj_num):
                self.V_record[i].append(self.ell(q_vals_list[i], prob))
                log_sum+=np.log(self.ell(q_vals_list[i], prob))

            # avg_reward = self.ell(qrvals, prob)
            # avg_utility = self.ell(qgvals, prob)

            self.acc_avg_gap +=  log_sum
            # self.model.objective_value - avg_reward
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.gap.append(log_sum)
            self.avg_gap.append(self.acc_avg_gap / (self.iter_num))
            # self.V1.append(avg_reward)
            # self.V2.append(avg_utility)

    def calculate_G(self, num=None, verbose=False):
        
        # self.iter_num += 1
        if verbose: print("Calculate G of iteration:", self.iter_num+1)

        prob = self.theta_to_policy()    
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        # Calculate for all objectives
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))
        # Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[0]))
        # Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))

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
        mat = np.identity(self.s * self.A) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        # V(s): |S|*1
        # Calculate for all objectives
        V_list = []
        for i in range(self.obj_num):
            V_list.append(np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.rewards[i])))

        total_V = np.zeros(V_list[0].shape)
        for i in range(self.obj_num):
            total_V += V_list[i]*direction[i]
        # V = Vr / vrvals + Vg / vgvals
        vvals = np.dot(np.transpose(total_V), self.rho)

        q_vals_list = []
        for i in range(self.obj_num):
            q_vals_list.append(self.Q_cal(V_list[i], self.rewards[i]))

        global_reward = np.zeros(self.rewards[0].shape)
        for i in range(self.obj_num):
            global_reward += self.rewards[i]*direction[i]
        qvals = self.Q_cal(total_V, global_reward)

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        self.theta += self.step * naturalgradient

        if self.iter_num % self.div_number == 0:

            log_sum = 0
            for i in range(self.obj_num):
                self.V_record[i].append(self.ell(q_vals_list[i], prob))
                log_sum+=np.log(self.ell(q_vals_list[i], prob))

            self.acc_avg_gap +=  log_sum
            
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
            self.gap.append(log_sum)
            self.avg_gap.append(self.acc_avg_gap / (self.iter_num))
