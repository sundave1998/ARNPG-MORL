import numpy as np


class CMDP_NPG:
    def __init__(self, reward, utility, s, a, b, rho, gamma, prob_transition, model):
        self.reward = reward
        self.utility = utility
        self.s = s
        self.a = a
        self.b = b
        self.rho = rho
        self.gamma = gamma
        self.prob_transition = prob_transition
        self.model = model
        
        self.theta = np.random.uniform(0, 1, size=s * a)
        self.dual = 0
        self.gap = []
        self.violation = []
        self.acc_avg_gap = 0
        self.acc_avg_violation = 0
        self.div_number = 1  # 000
        self.step = 1
        self.dualstep = 1
        
        
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


    def proj(self, scalar, offset = 100):
#         offset = 100
        if scalar < 0:
            scalar = 0

        if scalar > offset:
            scalar = offset

        return scalar

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


    # Run policy iteration to get the optimal policy and compute the constraint violation
    # Feasibility checking: negative constraint violation leads to the Slater condition
    def policy_iter(self, q_vals, s, a):
        new_policy = np.zeros(s * a)
        for i in range(s):
            idx = np.argmax(q_vals[i * a:(i + 1) * a])
            new_policy[i * a + idx] = 1

        return new_policy
    
    def NPG_step(self, verbose=False):
        
        self.iter_num += 1
        if verbose: print("iteration:", self.iter_num)
#         for k in range(N):
        prob = self.theta_to_policy()
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)

        qrvals = np.dot(np.linalg.inv(mat), self.reward)
        qgvals = np.dot(np.linalg.inv(mat), self.utility)
        qvals = qrvals + self.dual * qgvals

        vrvals = self.ell(qrvals, prob)
        vgvals = self.ell(qgvals, prob)
        vvals = vrvals + self.dual * vgvals

        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)
        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

#         # primal natural gradient ascent
#         # dual projected sub-gradient descent
        self.theta += self.step * naturalgradient
    
        self.dual = self.proj(self.dual - self.dualstep * (self.ell(qgvals, prob) - self.b))

        if self.iter_num % self.div_number == 0:
            avg_reward = self.ell(qrvals, prob)
            avg_violation = self.b - self.ell(qgvals, prob)
#             # acc_avg_gap = model.objective_value - avg_reward
#             # acc_avg_violation = avg_violation
#             # print('Average gap:', acc_avg_gap)
#             # print('Average violation:', acc_avg_violation)
#             # gap.append(acc_avg_gap)
#             # violation.append(acc_avg_violation)
            self.acc_avg_gap += self.model.objective_value - avg_reward
            self.acc_avg_violation += avg_violation
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
                print('Average volation:', self.acc_avg_violation / (self.iter_num))
            self.gap.append(self.acc_avg_gap / (self.iter_num))
            self.violation.append(self.acc_avg_violation / (self.iter_num))