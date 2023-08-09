
  

## Authors' Response to All (Discussion on $c$ and $\delta^*$):
We wholeheartedly thank all the reviewers for their time and their constructive feedback on our paper. The reviewers' comments provided us with great insights on how to increase clarity and reduce potential confusion about our paper. All the reviewers have raised similar concerns regarding our assumptions on $c > 0$ and $\delta^* > 0$ in the paper. Please note that reference numbers and line numbers are based on the supplementary material (MPG Main\&Appendix.pdf).

First, for simplicity, we recall the definitions of $c^k, c_K, c$ and $\delta^k, \delta_K, \delta^*$ in the potential game setting. For agent $i$ at iteration $k$, define $a_{i_p}^k \in \arg\max_{a_j \in \mathcal{A}_i} \bar{r}_i^k(a_{j}) =: \mathcal{A}_{i_p}^k$ and $a_{i_q}^k \in \arg\max_{a_j \in \mathcal{A}_i\backslash \mathcal{A}_{i_p}^k} \bar{r}_i^k(a_{j})$, where $\mathcal{A}_{i_p}^k$ denotes the set of the best possible actions for agent $i$ in the current state and $\bar{r}_i(a_i) = \mathbb{E}_{a_{-i} \sim \pi_{-i}}[r_i(a_i, a_{-i})]$. We define
$$c^k := \min_{i\in [n]} \sum_{a_j \in \mathcal{A}_{i_p}^k} \pi_i^k(a_j) \in (0, 1),\quad
\delta^k := \min_{i \in [n]} [\bar{r}_i^k(a_{i_p}^k) - \bar{r}_i^k(a_{i_q}^k)] \in (0, 1). $$

Additionally, we denote
$c_K := \min_{ k\in[K]} c^k; c := \inf_{K \to \infty} c_K; \delta_K := \min_{k\in[K]}\delta^k, \delta^* := \lim_{k \to \infty} \delta^k$.

- The effect of $c$ was discussed in our submission at l. 175-183. Our exact definition of c is indeed the same as the $c$ defined in [33], with similar motivation and justifications. Combining Lemma 2 in [33] and Proposition 3.1 in this paper, $c^k$ asymptotically converges to 1. Since $c^k > 0$ for any softmax policy parameterization ($\pi_{i}^k(a_i|s) > 0$) and $c^k$ asymptotically converges to 1, we have $c := \inf_k c^k > 0$. This can also be observed in Fig. 1a in the original paper. Therefore, we believe our assumption on $c$ is mild and has been used within the same context in prior work.

- The introduction of the suboptimality gap $\delta^k$ enables us to draw a crucial connection between the difference in the potential function and the NE-gap using Lemma 3.2. As stated in Section 3.3, the best rate without this assumption is $O({1/\sqrt{K}})$.

	We would also like to justify our introduction of the sub-optimality gap with reference to two separate lines of work. In single-agent RL, Khodadadian et al. [R1] were able to prove asymptotic geometric convergence with the introduction of "optimal advantage function gap $\Delta^k$". $\Delta^k$ is very similar to our definition of $\delta^k$. Additionally, the notion of sub-optimality gap, though different in formulation, is commonly used in the multi-armed bandit literature [R2]. In both works, the introduction of a sub-optimality gap greatly benefits the analysis. It also serves a similar purpose in our work.

	We agree with the reviewers that requiring $\delta^k > 0$ to hold for all iteration $k$ is indeed restrictive. However, we found that occasional $\delta^k \approx 0$ does not affect the global convergence rate. Motivated by this observation, we have proposed a theoretical relaxation with Cor. 3.6.1. Instead of requiring the suboptimality gap to be always non-zero, we only require the limit (existence of a limit is due to asymptotic convergence guarantee in [33]) to be non-zero, which is far from restrictive. Recalling our definition in Eq. (6), the suboptimality gap is defined as the performance gap between the best and second best classes of actions. Having a zero suboptimality gap implies that all actions of one agent belong to one class with the exact same expected reward values, which implies a zero NE-gap for the specific client.

  

We will revise and rephrase the related statements in order to reduce the potential confusion to readers in the final version. Additionally, we will incorporate the valuable suggestions concerning other related literature, and correct the typos and notation errors pointed out by various reviewers.

[33] Runyu Zhang, Jincheng Mei, Bo Dai, Dale Schuurmans, and Na Li. On the global convergence rates of decentralized softmax gradient play in markov potential games. Advances in Neural Information Processing Systems, 35:1923–1935, 2022.

[R1] Khodadadian, Sajad, et al. "On linear and super-linear convergence of Natural Policy Gradient algorithm." Systems \& Control Letters 164 (2022): 105214.

[R2] Lattimore, T. and Szepesvári, C., 2020. Bandit algorithms. Cambridge University Press.

# Response to gegG
We thank the reviewer for the valuable feedback regarding the paper. Please see our responses below with respect to the specific comments. We believe that we have addressed all the concerns raised by the reviewer, and we sincerely hope that the reviewer would consider increasing the score.  

**Q1.** "$c > 0, \delta > 0$ could be too strong and unnecessary."

**Response:** We understand the concern on $c$ and $\delta$ raised by the reviewer. **We refer to the above "Authors' Response to All" for a more detailed explanation.**

**Q2.** "... this justification is not suitable. In single-agent settings, NPG has a nice convergence that does not depend on $1/c$, while [17] shows that PG can take exponential many iterations to converge ... directly assuming $c>0$ is still too strong (and possibly unnecessary)."

**Response:**
In single-agent RL, policy gradient depend on a product of advantage function, occupancy measure, and the action probability (Eqn. 10 in [1]). Therefore, it is possible for PG algorithm to make a small update although the advantage function is significant. Single-agent NPG solved this issue using the Moore-Penrose inverse of Fisher information matrix to cancel out occupancy measure as well as action probability. However, in MARL, Fisher information matrix does not fully cancel out everything, since its calculation only uses local policy. Based on this observation, we make the comparison between single-agent PG and Multi-agent NPG.

Note that we are not the first work to introduce $1/c$ in the analysis of multi-agent independent NPG algorithms. The same assumption is made in [33] with similar accompanying statements saying that "Based on our analysis and numerical results, even for natural gradient play—which is known to enjoy dimension free convergence in single agent learning we find in the multiagent setting that it can still become stuck in these undesirable regions. Such evidence suggests that preconditioning according to the Fisher information matrix is not sufficient to ensure fast convergence in multi-agent learning. "

We will rephrase our statements accordingly in the final version.

[1] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. The Journal of Machine Learning Research, 22(1):4431–4506, 2021.

[33] Runyu Zhang, Jincheng Mei, Bo Dai, Dale Schuurmans, and Na Li. On the global convergence rates of decentralized softmax gradient play in markov potential games. Advances in Neural Information Processing Systems, 35:1923–1935, 2022.

  
  

# Response to Qt2L

We thank the reviewer for the time and support of our paper as well as the valuable suggestions. We are encouraged by the fact that the reviewer finds our paper "nicely written", with "nice convergence rate results", andthat it is "self-contained". Please see our response below with respect to the specific comments.

**Q1.** "$c$ and $\delta_K$ are not controlled. They could be arbitrarily small, at least without further work."

**Response:** We understand the concern on $c$ and $\delta_K$ raised by the reviewer. **We refer to the above "Authors' Response to All" for a more detailed explanation.**


**Q2.** "... the step up to Markovian settings is much less than in a less restrictive setting."

**Response:** We agree that the Markov potential game (MPG) is a restrictive setting compared to the general multi-agent RL (MARL) setting. Considering the difficulty of multi-agent settings, we believe that the convergence analysis of independent NPG in MPG will be an important step in solving the general MARL problems.

**Q3.** "A more philosophical point is that the article assumes players can receive oracle information. ... where learners might be able to access the advantage functions that are required to implement the method."

**Response:** We follow [33] and only consider the oracle setting in our analysis. In general, an oracle can be estimated by Monte Carlo or temporal difference methods for the stochastic setting and can be analyzed similarly as [1]. However, as mentioned by the Reviewer XDXN, it is much harder to handle $c$ in the stochastic setting. We will add the above comment and mention the sample-based results in [8] in our final version. We leave the related analysis as future work.


**Q4.** "... and I would expect the authors to compare their results with those presented under the multiplicative weights description."


**Response:** As pointed out by [1] (after Lemma 15), NPG with softmax parameterization is "identical to the classical multiplicative weights updates" in the single-agent case.


**Q5.** "function f ... also important dependence on k, and on $\pi_{-i}^k$."


**Response:** At iteration $k$, both $\pi_i^k$ and $\bar{r}_i^k$ (oracle) are known and fixed. So the only decision variable for function $f$ is $\alpha$. For clarity, we will use $f^k(\alpha)$ in the final version.


**Q6.** "the definition of $a^k_{i_q}$ is clumsy ... ."

**Response:**

For the main paper, it is enough to only define $\delta^k$ directly. The definitions of $a^k_{i_q}$ and $a^k_{i_p}$ were in fact used only for the proof of Lemma 3.2 in the appendix. We have included them in our main paper only for consistency of analysis.


**Q7.** "... we begin to get bitten by the lack of care in defining what the functions depend on"


**Response:**

Please refer to the response to Q5.


**Q8.** "... Furthermore, it seems strange to take a limit of $c_K$ values, but not take a limit of $\delta_K$ values in the theory that follows"


**Response:**

$c^k > 0$ and $c > 0$ are guaranteed as shown in [33], but $\delta^k > 0$ is not necessarily true for all iterations. For a few iterations $k$, it is possible to have $\delta^k = 0$. Therefore, we only define $\delta^* = \lim_{k \to \infty} \delta^k$ and use $\delta^*$ in the upper bound (cf. Table 1 and Corollary 3.6.1).


**Q9.** "First inequality in the proof of Theorem 3.3, I think should be an equality?"


**Response:**

It should be an inequality due to the fact that NE-gap takes the maximum over all agents, whereas the function $f$ is defined with respect to the summation.


**Q10.** "Line 259, I think we cannot claim that $|\delta^k-\delta^*|$ is small, since we are only assuming a lim inf?"


**Response:**

The reviewer is correct. In fact, asymptotic convergence for this algorithm is guaranteed in previous works [33], and a limit exists. We will replace $\liminf$ by $\lim$ in the final version.


[1] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. The Journal of Machine Learning Research, 22(1):4431–4506, 2021.


[8] Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, and Mihailo Jovanovic. Independent policy gradient for large-scale markov potential games: Sharper rates, function approximation, and game-agnostic convergence. In International Conference on Machine Learning, pages 5166–5220. PMLR, 2022.

[33] Runyu Zhang, Jincheng Mei, Bo Dai, Dale Schuurmans, and Na Li. On the global convergence rates of decentralized softmax gradient play in markov potential games. Advances in Neural Information Processing Systems, 35:1923–1935, 2022.



# Response to tti4


We thank the reviewer for the positive comments and the suggestions concerning our paper. Please see our response below with respect to the specific comments.


**Q1.** "The new results do not seem to be directly comparable to the previous ones due to the use of a suboptimality gap. I encourage the authors to explicitly state this early in the paper to avoid confusion."


**Response:**
We use Table 1 to provide a clear summary for our result and existing works, which performs the same function as Table 1 in [33]. $\delta^*$ was included in the iteration complexity in our Table 1.
We will further clarify the use of the suboptimality gap in our statements in the abstract and contributions.


**Q2.** "I also encourage the authors to complement the work with more empirical results."


**Response:**
We have added some additional material and purposefully constructed example tasks to show the impact of $\delta^*$ on the algorithm in practice.

We consider an example of a 2-by-2 matrix game with the reward matrix
$r = \begin{bmatrix}
1&2\\3+\delta^*&3
\end{bmatrix}$.

For the experiments, we have selected various values of $\delta^*$ ranging from $1e^{-3}$ to 10. We run the same algorithm with the same initial policy for all experiments, and plot both the NE-gap and the L1 accuracy of the algorithm. It can be seen from the experiments that $\delta^*$ indeed plays an important role in the convergence of the algorithm.

**Please refer to newly attached pdf in "Authors' Response to All" for details.**


**Q3.** "It may also be helpful to state when the suboptimality gap is very small, the results can degenerate into previous results."


**Response:**
In fact, the iteration complexity of independent NPG algorithms is the smaller of the results in this paper and those in [33]. The specific minimum value depends on $\delta^*$, which depends on the structure of (Markov) potential games.

  

**Q4.** "Is the c defined as the same as this paper?"


**Response:** Yes, the definition of $c$ is the same.


**Q5.** "Will log barrier regularization have the same effect on the results presented in this paper?"


**Response:** Since the log-barrier regularization repels the trajectory from regions with small policy values, we can derive a lower bound for policy value as Lemma 24 in [33]. While it is true that the log-barrier regularization can remove the dependence on $1/c$, the introduction of log-barrier parameter $\lambda$ makes the convergence rate slower by $O(1/\sqrt{K})$ since the upper bound of convergence rate has the form $\frac{c_1}{\lambda K} + c_2 \lambda$.


**Q6.** "In Definition 2.1, the potential function is assumed to take a specific form with $\phi$. However, this form does not seem to be needed in previous analyses. How important is this specific form to the analysis and can it be lifted?"


**Response:**
Our formulation of MPG was originally adapted from that of [33, 34, R1, R2]. The analysis will become harder without the definition of $\phi$, since this definition provides critical additional structure to the problem.


[33] Runyu Zhang, Jincheng Mei, Bo Dai, Dale Schuurmans, and Na Li. On the global convergence rates of decentralized softmax gradient play in markov potential games. Advances in Neural Information Processing Systems, 35:1923–1935, 2022.


[34] Runyu Zhang, Zhaolin Ren, and Na Li. Gradient play in multi-agent markov stochastic games: Stationary points and convergence. arXiv preprint arXiv:2106.00198, 2021.


[R1] Macua, Sergio Valcarcel, Javier Zazo, and Santiago Zazo. "Learning Parametric Closed-Loop Policies for Markov Potential Games." International Conference on Learning Representations. 2018.


[R2] Zazo, Santiago, et al. "Dynamic potential games with constraints: Fundamentals and applications in communications." IEEE Transactions on Signal Processing 64.14 (2016): 3806-3821.

  
  

# Response to XDXN

We thank the reviewer for the detailed assessment and constructive feedback on the paper, with both main paper and appendix. We are encouraged by the fact that the reviewer finds our paper "well-organized" with "new and interesting analysis". Our submission has been revised to include previously missing relevant works, fix typos and rephrase potential obscurities raised by the reviewer. We also address the reviewer's concerns and questions below.

> About the suboptimality gap $\delta_K$ and the convergence rate: ... convergence is actually 'asymptotic'.

We would like to provide a more detailed discussion on the factor $\delta^k, \delta_K, \delta^*$ in Authors' Response to All. We will clarify the "asymptotic" property of results in abstract, contributions, and Table 1 in our final version.

> About Assumption 3.2 guarantees that ... hard to interpret although the 'sub-optimality gap’ is a standard quantity in the bandit literature....

The reviewer is correct that it is hard to guarantee a lower bound on $\delta^k, \forall k$. Therefore, we proposed the theoretical relaxation with Corollary 3.6.1 based on $\delta^*$. **Please see the details in Authors' Response to All.**

In the bandit literature, the suboptimality gap indicates the structure of the environment. In the multi-agent setting, other agents can also be seen as part of environment w.r.t. a specific agent. Based on this intuition, we introduce this concept based on the marginalized reward and use Lemma 3.2 to capture the impact of $\delta^k$.
> related works

We sincerely thank the reviewer for pointing out these previous works and we will include them in the final version.
> definition of MPGs

Our formulation of MPG was originally adapted from that of [33, 34, R1, R2]. We consider this formulation to be well-known and somewhat standard in the literature. We will mention this difference in formulation with additional statements alongside Table 1 as well as Section 2.
> Originality of the analysis

We thank the reviewer for acknowledgement of the novelty of our analysis in Lemmas 3.2, B.3, and C.1. The proofs of Lemmas A.1 and A.2 are similar to [4], but we arrive at different conclusions due to new lemmas (Lemmas 3.2 and B.3) as bridges. The analysis of Lemma B.2 is related to [33] but with some key differences. Firstly, we provide a new Lemma B.1 to better explain the final equality in Line 476, which doesn't appear in [33]. Secondly, we use Young's inequality in Line 479 to save a $\sqrt{n}$ factor compared with [33]. Nevertheless, we will make sure to mention these works while providing revised proofs of these lemmas.
> Clarity
- We use the definition of 'stationary point' from mathematics and optimization to denote a point with zero gradients. Incidentally, in our paper's context, the stationary point denotes where a set of policies have reached NE.
- Yes, we used 'set of policies' to denote a joint policy of product simplices. We will make sure to clarify at our first use of this specific term.
- We meant the quantities defined in l. 153 and we will clarify it in the final version.
- Yes, the definition of $f(\infty)$ will be included in our updated draft.
- Yes, $\mathcal{R}_i$ is the set of reward functions with a particular structure. The ambient space is $\mathbb{R}^{\prod_{j} |\mathcal{A}_j|}$. We wanted to provide a discussion over the structure of potential games in Lemma A.4 without further complicating other lemmas or theorems.
- Here our intention was to show for a vector $\mathbf{r} = [r_1, ..., r_n]$, where $r_1 > r_2 ... > r_n$. We will make sure to clarify this in our statements.
> Typos

We thank the reviewer for the detailed evaluation of the paper. We will fix these typos in our final version.
> 'isolated stationary policies’... as stated in Theorem 3.6 (and [33])... not stated in the theorem. How does the theorem guarantee that $c>0$?

Yes, this assumption is required. We will add it to the final version.
> ‘isolated stationary policies’. What do we mean by ‘stationary’ policies in this context? Although such a terminology is used in [33], it is not clear to the reader what this means, especially that ‘stationary’ has also another meaning for policies (time-independent).

A stationary policy is what our paper describes as a 'stationary point', a set of policies that jointly reaches NE. In this sense, 'isolated stationary policy' means no other stationary points exist in an open neighborhood of any stationary point. We will add this clarification in the revision.
> ... isn’t it possible to use the assumption to obtain corollary 3.6.1? ... would guarantee that Assumption 3.2 hold.

The mild Assumption 3.2 assumes the limit $\delta^*$ is larger than 0, which is required by Corollary 3.6.1. It is not possible to only use "stationary policies are isolated" to obtain Corollary 3.6.1.

A counter example would be a 2-by-2 matrix game with $r_{11} = 1, r_{12} = 1, r_{21} = 2, r_{22} = 1$, where one NE would be $\pi_1 = \{0, 1\}, \pi_2 = \{1, 0\}$. In this example, we have an isolated stationary policy with $\delta^* = 0.$
> precise the mathematical definition of $\pi_i^{*k}$?

The exact definition is that $\pi_i^{*k} \in \arg\max_{\pi_i} V_i^{\pi_i, \pi_{-i}^k}(\rho)$.

> l. 153: why $f(\eta)\geq 0$?

NPG updates $\pi_i^{k+1}$ as follows: $\pi_i^{k+1}(a_i|s) = \arg\max_{\pi_i} \eta \langle \bar{r}_i^k(\cdot), \pi_i(\cdot) \rangle - KL(\pi_i || \pi_i^k)$. Therefore, $f(\eta) = \sum_i \langle \bar{r}_i^k(\cdot), \pi_i^{k+1}(\cdot) - \pi_i^{k}(\cdot) \rangle \geq \sum_i KL(\pi_i^{k+1} || \pi_i^k) / \eta \geq 0$.
> Extension to the stochastic setting:

We follow [33] and only consider the oracle setting in our analysis. In general, an oracle can be estimated by Monte Carlo or temporal difference methods for the stochastic setting and can be analyzed similarly as [1]. However, as mentioned by the reviewer, it is much harder to handle $c$ in the stochastic setting. We will add the above comment and mention the sample-based results in [8] in our final version. We leave the related analysis as future work.

*For brevity, please see our response to reviewer Qt2L and tti4 for details of referenced works.*



# shorter version below 6000 char:

We thank the reviewer for the detailed assessment and constructive feedback on the paper, with both main paper and appendix. We are encouraged by the fact that the reviewer finds our paper "well-organized" with "new and interesting analysis". Our submission has been revised to include previously missing relevant works, fix typos and rephrase potential obscurities raised by the reviewer. We also address the reviewer's concerns and questions below.

> About $\delta_K$ and rate... convergence is actually 'asymptotic'.

A more detailed discussion on $\delta^k, \delta_K, \delta^*$ is provided **in Authors' Response to All**. We will clarify the "asymptotic" property of results in abstract, contributions, and Table 1 in our final version.

> Asm. 3.2 guarantees... hard to interpret although... standard quantity in the bandit literature....

The reviewer is correct that it is hard to guarantee a lower bound on $\delta^k, \forall k$. Therefore, we proposed the theoretical relaxation with Cor. 3.6.1 based on $\delta^*$. **Please see the details in Authors' Response to All.**

In the bandit literature, the suboptimality gap indicates the structure of the environment. In the multi-agent setting, other agents can also be seen as part of environment w.r.t. a specific agent. Based on this intuition, we introduce this concept based on the marginalized reward and use Lem. 3.2 to capture the impact of $\delta^k$.
> related works

We sincerely thank the reviewer for pointing out these previous works and we will include them in the final version.
> definition of MPGs

Our formulation of MPG was originally adapted from that of [33, 34, R1, R2]. We consider this formulation to be well-known and somewhat standard in the literature. We will mention this difference in formulation with additional statements alongside Tab. 1 and Sec. 2.
> Originality of the analysis

We thank the reviewer for acknowledgement of the novelty of our analysis in Lem. 3.2, B.3, and C.1. The proofs of Lem. A.1 and A.2 are similar to [4], but we arrive at different conclusions due to new lemmas (Lem. 3.2 and B.3) as bridges. The analysis of Lem. B.2 is related to [33] but with some key differences. Firstly, we provide a new Lem. B.1 to better explain the final equality in l. 476, which doesn't appear in [33]. Secondly, we use Young's inequality in l. 479 to save a $\sqrt{n}$ factor compared with [33]. Nevertheless, we will make sure to mention these works while providing revised proofs of these lemmas.
> Clarity
- We use the definition of 'stationary point' from mathematics and optimization to denote a point with zero gradients. Incidentally, in our paper's context, the stationary point denotes where a set of policies have reached NE.
- Yes, we used 'set of policies' to denote a joint policy of product simplices. We will make sure to clarify at our first use of this specific term.
- We meant the quantities defined in l. 153 and we will clarify it in the final version.
- Yes, the definition of $f(\infty)$ will be included in our updated draft.
- Yes, $\mathcal{R}\_i$ is the set of reward functions with a particular structure. The ambient space is $\mathbb{R}^{\prod\_{j} |\mathcal{A}\_j|}$. We wanted to provide a discussion over the structure of PGs in Lem. A.4 without further complicating other lemmas or theorems.
- Here our intention was to show for a vector $\mathbf{r} = [r_1, ..., r_n]$, where $r_1 > r_2 ... > r_n$. We will make sure to clarify this in our statements.

> Typos

We thank the reviewer for the detailed evaluation of the paper. We will fix these typos in our final version.
> 'isolated stationary policies’... as stated in Thm. 3.6 (and [33])... not stated in the theorem. How does the theorem guarantee that $c>0$?

Yes, this assumption is required. We will add it to the final version.
> ‘isolated stationary policies’. What do we mean by ‘stationary’ policies in this context? Although such a terminology is used in [33], it is not clear to the reader what this means, especially that ‘stationary’ has also another meaning for policies (time-independent).

A stationary policy is what our paper describes as a 'stationary point', a set of policies that jointly reaches NE. In this sense, 'isolated stationary policy' means no other stationary points exist in an open neighborhood of any stationary point. We will add this clarification in the revision.
> ... isn’t it possible to use the assumption to obtain corollary 3.6.1? ... would guarantee that Assumption 3.2 hold.

The mild Asm. 3.2 assumes the limit $\delta^*$ is larger than 0, which is required by Cor. 3.6.1. It is not possible to only use "stationary policies are isolated" to obtain Cor. 3.6.1.

A counter example would be a 2-by-2 matrix game with $r_{11} = 1, r_{12} = 1, r_{21} = 2, r_{22} = 1$, where one NE would be $\pi_1 = \{0, 1\}, \pi_2 = \{1, 0\}$. In this example, we have an isolated stationary policy with $\delta^* = 0.$
> precise definition of $\pi_i^{*k}$?

The exact definition is that $\pi_i^{*k} \in \arg\max_{\pi_i} V_i^{\pi_i, \pi_{-i}^k}(\rho)$.

> l. 153: why $f(\eta)\geq 0$?

NPG updates $\pi_i^{k+1}$ as follows: $\pi_i^{k+1}(a_i|s) = \arg\max_{\pi_i} \eta \langle \bar{r}_i^k(\cdot), \pi_i(\cdot) \rangle - KL(\pi_i || \pi_i^k)$. Therefore, $f(\eta) = \sum_i \langle \bar{r}_i^k(\cdot), \pi_i^{k+1}(\cdot) - \pi_i^{k}(\cdot) \rangle \geq \sum_i KL(\pi_i^{k+1} || \pi_i^k) / \eta \geq 0$.
> Extension to the stochastic setting:

We follow [33] and only consider the oracle setting in our analysis. In general, an oracle can be estimated by Monte Carlo or temporal difference methods for the stochastic setting and can be analyzed similarly as [1]. However, as mentioned by the reviewer, it is much harder to handle $c$ in the stochastic setting. We will add the above comment and mention the sample-based results in [8] in our final version. We leave the related analysis as future work.

*For brevity, see response to reviewers Qt2L and tti4 for referenced works.*