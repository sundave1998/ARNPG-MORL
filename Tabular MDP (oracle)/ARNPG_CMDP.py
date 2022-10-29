import numpy as np
# from docplex.mp.model import Model
from NPG_CMDP import CMDP_NPG


class CMDP_ARNPG(CMDP_NPG):
    def __init__(self, reward, utility, s, a, b, rho, gamma, prob_transition, model):
        super().__init__(reward, utility, s, a, b, rho, gamma, prob_transition, model)
        self.alpha = 0.2
        self.inner = 1

        
    def naturalgradient(self, old_prob, old_vg):
        prob = self.theta_to_policy()
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)

        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)

        Vr = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.reward))
        Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))

        vrvals = np.dot(np.transpose(Vr), self.rho)
        vgvals = np.dot(np.transpose(Vg), self.rho)
        vvals = vrvals + (self.dual - self.dualstep * (old_vg - self.b)) * vgvals

        d_s_pi = np.linalg.inv(np.identity(self.s) - self.gamma * P_theta)  # |S|*|S|
        kl_divergence = np.sum(prob.reshape(self.s, self.a) * np.log(prob.reshape(self.s, self.a) / old_prob.reshape(self.s, self.a)), axis=1)
        regular_term = np.dot(d_s_pi, kl_divergence)
        V = Vr + (self.dual - self.dualstep * (old_vg - self.b)) * Vg - self.alpha / (1 - self.gamma) * regular_term
        qvals = self.Q_cal(V, self.reward + (self.dual - self.dualstep * (old_vg - self.b)) * self.utility + self.alpha * np.log(old_prob))

        MPinverse = np.linalg.pinv(self.Fisher_info(prob, d_pi))
        gradient = self.grad(qvals - vvals - self.alpha * np.log(prob), prob, d_pi)
        return np.matmul(MPinverse, gradient)
    
    def ARNPG_step(self, verbose = False):
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
        vvals = vrvals + (self.dual - self.dualstep * (vgvals - self.b)) * vgvals

        qrvals = self.Q_cal(Vr, self.reward)
        qgvals = self.Q_cal(Vg, self.utility)
        qvals = qrvals + (self.dual - self.dualstep * (vgvals - self.b)) * qgvals

        if self.iter_num % self.div_number == 0:
            avg_reward = self.ell(qrvals, prob)
            avg_violation = self.b - self.ell(qgvals, prob)
            # acc_avg_gap = model.objective_value - avg_reward
            # acc_avg_violation = avg_violation
            # print('Average gap:', acc_avg_gap)
            # print('Average violation:', acc_avg_violation)
            # gap.append(acc_avg_gap)
            # violation.append(acc_avg_violation)
            self.acc_avg_gap += self.model.objective_value - avg_reward
            self.acc_avg_violation += avg_violation
            if verbose:
                print('Average gap:', self.acc_avg_gap / (self.iter_num))
                print('Average violation:', self.acc_avg_violation / (self.iter_num))
            self.gap.append(self.acc_avg_gap / (self.iter_num))
            self.violation.append(self.acc_avg_violation / (self.iter_num))

        old_prob = prob
        for l in range(self.inner):
            ng = self.naturalgradient(old_prob, vgvals)
            self.theta += self.step * ng

        prob = self.theta_to_policy()
        Pi = self.get_Pi(prob)
        mat = np.identity(self.s * self.a) - self.gamma * np.matmul(self.prob_transition, Pi)
        P_theta = np.matmul(Pi, self.prob_transition)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)
        d_pi = (1 - self.gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(self.s) - self.gamma * P_theta))), self.rho)
        
        Vg = np.dot(np.linalg.inv(np.identity(self.s) - self.gamma * P_theta), np.matmul(Pi, self.utility))
        qgvals = self.Q_cal(Vg, self.utility)
        self.dual = max(self.dual - self.dualstep * (self.ell(qgvals, prob) - self.b), self.dualstep * (self.ell(qgvals, prob) - self.b))




        