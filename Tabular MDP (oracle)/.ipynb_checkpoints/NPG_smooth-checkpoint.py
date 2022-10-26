import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import copy


"""Natural Policy Gradient Method with Softmax Parametrization
"""

# Random Seed
np.random.seed(10)
# Problem Setup
gamma = 0.8
s, a = 20, 10

# Randomly generated probability transition matrix P((s,a) -> s') in [0,1]^{|S||A| x |S|}
raw_transition = np.random.uniform(0, 1, size=(s * a, s))
prob_transition = raw_transition / raw_transition.sum(axis=1, keepdims=1)
# Random positive rewards
reward = np.random.uniform(0, 1, size=(s * a))
# Random positive utilities
utility = np.random.uniform(0, 1, size=(s * a))
# Start state distribution
rho = np.ones(s) / s


def theta_to_policy(theta, s, a):
    """
    :param theta: |S||A| * 1
    :param s: |S|
    :param a: |A|
    :return: |S||A| * 1
    """
    prob = []
    for i in range(s):
        norm = np.sum(np.exp(theta[a * i:a * (i + 1)]))
        for j in range(a * i, a * (i + 1)):
            prob.append(np.exp(theta[j]) / norm)

    return np.asarray(prob)


def get_Pi(prob, s, a):
    """
    :param prob: |S||A| * 1
    :param s: |S|
    :param a: |A|
    :return: |S| * |S||A|
    """
    Pi = np.zeros((s, s * a))
    for i in range(s):
        Pi[i, i * a:(i + 1) * a] = prob[i * a:(i + 1) * a]

    return Pi


def grad_state_action(prob, state, action):
    """
    :param prob: |S||A| * 1
    :param state: 1 * 1
    :param action: 1 * 1
    :return: \nabla_{\theta} \pi_{\theta}(s,a)
    """
    grad = np.zeros(s * a)
    for j in range(0, a):
        if j == action:
            grad[a * state + j] = prob[a * state + j] * (1 - prob[a * state + j])
        else:
            grad[a * state + j] = -prob[a * state + action] * prob[a * state + j]

    return grad


def grad_state(qvals, prob, state):
    grad = np.sum([qvals[state * a + i] * grad_state_action(prob, state, i) for i in range(0, a)], axis=0)
    return grad


def grad(qvals, prob, d_pi):
    grad = np.sum([d_pi[i] * grad_state(qvals, prob, i) for i in range(0, s)], axis=0)
    return grad


def Fisher_info(prob, d_pi):
    """
    :param prob: |S||A * 1
    :param d_pi: |S| * 1
    :return: Fisher information matrix \nabla_{\theta} \pi_{\theta}(s,a) x {\nabla_{\theta} \pi_{\theta}(s,a)}^T
    """
    qvals_one = np.ones(s * a)
    grad = np.sum([d_pi[i] * grad_state(qvals_one, prob, i) for i in range(0, s)], axis=0)
    fisher = np.outer(grad, grad) + 1e-3 * np.identity(s * a)
    return fisher


def ell(qvals, prob, rho):
    """
    Calculate V from Q value function
    :param qvals: |S||A| * 1
    :param prob: |S||A| * 1
    :param rho: |S| * 1
    :return: V |S| * 1
    """
    V = np.zeros(s)
    for i in range(s):
        V[i] = np.sum([qvals[i * a + j] * prob[i * a + j] for j in range(a)])

    ell = np.dot(V, rho)
    return ell


def Q_cal(V, func):
    """
    Calculate Q from V value function
    :param V: |S| * 1
    :param func: reward/cost function |S||A| * 1
    :return: Q |S||A| * 1
    """
    Q = np.zeros(s * a)
    for i in range(s):
        for j in range(a):
            Q[i * a + j] = func[i * a + j] + gamma * np.matmul(prob_transition[i * a + j], V)
    return Q


# calculate the optimal reward via QP
model = Model()
# create continuous variables
idx = [(i, j) for i in range(s) for j in range(a)]
x = model.continuous_var_dict(idx)

for i in range(s):
    for j in range(a):
        model.add_constraint(x[i, j] >= 0)

for s_next in range(s):
    model.add_constraint(
        gamma * model.sum(x[i, j] * prob_transition[i * a + j][s_next] for i in range(s) for j in range(a))
        + (1 - gamma) * rho[s_next] == model.sum(x[s_next, a_next] for a_next in range(a)))

model.maximize(model.sum(x[i, j] * reward[i * a + j] / (1 - gamma) for i in range(s) for j in range(a)) *
               model.sum(x[i, j] * utility[i * a + j] / (1 - gamma) for i in range(s) for j in range(a)))
# search for a globally optimal solution to a nonconvex model
model.parameters.optimalitytarget = 3
model.solve()
print(np.log(model.objective_value))


# Natural Policy Gradient Method with Softmax Parametrization
N = 300
theta0 = np.random.uniform(0, 1, size=s * a)
gap = []
div_number = 1
steps = [4.5]
final_gap = []
V1 = []
V2 = []

for step in steps:
    theta = copy.deepcopy(theta0)
    acc_avg_gap = 0
    for k in range(N):
        prob = theta_to_policy(theta, s, a)
        Pi = get_Pi(prob, s, a)
        mat = np.identity(s * a) - gamma * np.matmul(prob_transition, Pi)
        P_theta = np.matmul(Pi, prob_transition)
        d_pi = (1 - gamma) * np.dot(np.transpose((np.linalg.inv(np.identity(s) - gamma * P_theta))), rho)

        # V(s): |S|*1
        Vr = np.dot(np.linalg.inv(np.identity(s) - gamma * P_theta), np.matmul(Pi, reward))
        Vg = np.dot(np.linalg.inv(np.identity(s) - gamma * P_theta), np.matmul(Pi, utility))
        vrvals = np.dot(np.transpose(Vr), rho)
        vgvals = np.dot(np.transpose(Vg), rho)

        V = Vr / vrvals + Vg / vgvals
        vvals = np.dot(np.transpose(V), rho)

        qrvals = Q_cal(Vr, reward)
        qgvals = Q_cal(Vg, utility)
        qvals = Q_cal(V, reward / vrvals + utility / vgvals)

        MPinverse = np.linalg.pinv(Fisher_info(prob, d_pi))
        gradient = grad(qvals - vvals, prob, d_pi)
        naturalgradient = np.matmul(MPinverse, gradient)

        # primal natural gradient ascent
        # dual projected sub-gradient descent
        theta += step * naturalgradient

        if k % div_number == 0:
            avg_reward = ell(qrvals, prob, rho)
            avg_utility = ell(qgvals, prob, rho)
            acc_avg_gap += np.log(model.objective_value) - np.log(avg_reward*avg_utility)
            print('Average gap:', acc_avg_gap / (k + 1))
            gap.append(acc_avg_gap / (k + 1))
            V1.append(avg_reward)
            V2.append(avg_utility)
    final_gap.append(gap[-1])


# Saving the data. This can be loaded to make the figure again.
np.savetxt('NPG_gap_s20a10g8b3.txt', gap)