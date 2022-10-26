import numpy as np
from MDP import MDP

class CMDP_NPG(MDP):
    def __init__(self, reward, utility, s, a, b, rho, gamma, prob_transition, model):
        super().__init__(reward, utility, s, a, rho, gamma, prob_transition, model)
        self.b = b
        self.div_number = 1  # 000
        self.step = 1
        self.dualstep = 1
        self.violation = []
        self.acc_avg_violation = 0



    def proj(self, scalar, offset = 100):
#         offset = 100
        if scalar < 0:
            scalar = 0

        if scalar > offset:
            scalar = offset

        return scalar

#     def Q_cal(self, V, func):
#         """
#         Calculate Q from V value function
#         :param V: |S| * 1
#         :param func: reward/cost function |S||A| * 1
#         :return: Q |S||A| * 1
#         """
#         Q = np.zeros(self.s * self.a)
#         for i in range(self.s):
#             for j in range(self.a):
#                 Q[i * self.a + j] = func[i * self.a + j] + self.gamma * np.matmul(self.prob_transition[i * self.a + j], V)
#         return Q


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