import numpy as np
import collections

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Warning: gym or gymnasium not found. Using placeholder spaces classes.")

    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def sample(self):
                return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class gym:
        class Env:
            pass


class InvManagementEnv(gym.Env):
    """
    InvManagement-v1 (Lost Sales) implementation based on or-gym source code.
    Re-implemented to be compatible with modern gymnasium and strictly follow
    the logic from hubbs5/or-gym/envs/supply_chain/inventory_management.py
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 periods=30,
                 I0=[100, 100, 200],
                 p=2.0,
                 r=[1.5, 1.0, 0.75, 0.5],
                 k=[0.10, 0.075, 0.05, 0.025],
                 h=[0.15, 0.10, 0.05],
                 c=[100, 90, 80],
                 L=[3, 5, 10],
                 dist=1,
                 dist_param={'mu': 20},
                 alpha=0.97,
                 seed_int=None,
                 max_steps_per_episode=30,
                 render_mode=None):

        super().__init__()

        self.periods = periods
        self.I0 = I0
        self.p = p
        self.r = r
        self.k = k
        self.h = h
        self.c = c
        self.L = L
        self.backlog = False
        self.dist = dist
        self.dist_param = dist_param
        self.alpha = alpha
        self.seed_int = seed_int
        self.render_mode = render_mode
        self.max_steps_per_episode = periods

        self.init_inv = np.array(list(self.I0))
        self.num_periods = self.periods
        self.unit_price = np.append(self.p, self.r[:-1])
        self.unit_cost = np.array(self.r)
        self.demand_cost = np.array(self.k)
        self.holding_cost = np.append(self.h, 0)
        self.supply_capacity = np.array(list(self.c))
        self.lead_time = np.array(list(self.L))
        self.discount = self.alpha

        self.num_stages = len(self.init_inv) + 1
        m = self.num_stages
        lt_max = self.lead_time.max()

        self.action_space = spaces.Box(
            low=np.zeros(m-1),
            high=self.supply_capacity.astype(np.float32),
            dtype=np.float32
        )

        self.pipeline_length = (m-1) * (lt_max + 1)

        self.observation_space = spaces.Box(
            low=np.zeros(self.pipeline_length),
            high=np.ones(self.pipeline_length) *
            self.supply_capacity.max() * self.num_periods,
            dtype=np.float32
        )

        self.reset(seed=self.seed_int)

    def _update_state(self):
        """
        Replicates _update_state from or-gym.
        State vector = [Current Inventory (M-1) ... Action History (M-1 * Lt_max) ... ]
        Actually, the or-gym logic is:
        state size = m * (lt_max + 1) where m is (num_stages - 1).

        It fills:
        Index 0 to m-1: Current Inventory
        Index m to end: Recent action history (pipeline proxy)
        """
        m = self.num_stages - 1
        t = self.period
        lt_max = self.lead_time.max()

        state = np.zeros(m * (lt_max + 1), dtype=np.int32)

        if t == 0:
            state[:m] = self.I0
        else:
            state[:m] = self.I[t]

        if t == 0:
            pass
        elif t >= lt_max:
            state[-m*lt_max:] += self.action_log[t-lt_max:t].flatten()
        else:
            state[-m*t:] += self.action_log[:t].flatten()

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        periods = self.num_periods
        m = self.num_stages

        self.I = np.zeros([periods + 1, m - 1])
        self.T = np.zeros([periods + 1, m - 1])
        self.R = np.zeros([periods, m - 1])
        self.D = np.zeros(periods)
        self.S = np.zeros([periods, m])
        self.B = np.zeros([periods, m])
        self.LS = np.zeros([periods, m])
        self.P = np.zeros(periods)

        self.period = 0
        self.I[0, :] = np.array(self.init_inv)
        self.T[0, :] = np.zeros(m - 1)
        self.action_log = np.zeros((periods, m - 1), dtype=np.int32)

        obs = self._update_state()
        return obs, {}

    def step(self, action):
        """
        Replicates _STEP from or-gym.
        """
        action = np.array(action).reshape(-1)
        R = np.maximum(action, 0).astype(int)

        n = self.period
        L = self.lead_time
        I = self.I[n, :].copy()
        T = self.T[n, :].copy()
        m = self.num_stages
        c = self.supply_capacity

        self.action_log[n] = R.copy()

        Im1 = np.append(I[1:], np.inf)


        Rcopy = R.copy()

        mask_cap = R >= c
        R[mask_cap] = c[mask_cap]

        mask_inv = R >= Im1
        R[mask_inv] = Im1[mask_inv]

        self.R[n, :] = R

        RnL = np.zeros(m-1)
        for i in range(m-1):
            if n - L[i] >= 0:
                RnL[i] = self.R[n - L[i], i].copy()
                I[i] = I[i] + RnL[i]

        D0 = np.random.poisson(self.dist_param['mu'])
        self.D[n] = D0
        D = D0


        S0 = min(I[0], D)
        S = np.append(S0, R)
        self.S[n, :] = S

        I = I - S[:-1]
        T = T - RnL + R

        self.I[n+1, :] = I
        self.T[n+1, :] = T

        U = np.append(D, Rcopy) - S

        LS = U
        self.LS[n, :] = LS
        self.B[n, :] = 0

        p = self.unit_price
        r = self.unit_cost
        k = self.demand_cost
        h = self.holding_cost
        a = self.discount

        II = np.append(I, 0)
        RR = np.append(R, S[-1])

        profit = a**n * np.sum(p*S - (r*RR + k*U + h*II))
        self.P[n] = profit

        self.period += 1
        obs = self._update_state()
        done = self.period >= self.num_periods

        info = {
            "demand": D0,
            "sales": S,
            "lost_sales": LS,
            "inventory": I
        }

        return obs, profit, done, False, info

    def render(self):
        print(f"Step: {self.period}")
        print(f"Inventory: {self.I[self.period]}")
        print(f"Orders: {self.R[self.period-1] if self.period > 0 else 0}")
        print(f"Profit: {self.P[self.period-1] if self.period > 0 else 0}")
