import numpy as np
import collections

# Try to import gymnasium, otherwise fallback to gym or placeholders
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
                 dist=1,  # Poisson
                 dist_param={'mu': 20},
                 alpha=0.97,
                 seed_int=None,  # Changed from 0 to None for random initialization
                 max_steps_per_episode=30,  # Added for compatibility with my previous wrapper
                 render_mode=None):

        super().__init__()

        # --- Parameter Setup (Matching or-gym) ---
        self.periods = periods
        self.I0 = I0
        self.p = p
        self.r = r
        self.k = k
        self.h = h
        self.c = c
        self.L = L
        self.backlog = False  # v1 is Lost Sales
        self.dist = dist
        self.dist_param = dist_param
        self.alpha = alpha
        self.seed_int = seed_int
        self.render_mode = render_mode
        self.max_steps_per_episode = periods  # Alias

        # Process parameters into numpy arrays
        self.init_inv = np.array(list(self.I0))
        self.num_periods = self.periods
        # Price at stage m is Cost at stage m-1
        self.unit_price = np.append(self.p, self.r[:-1])
        self.unit_cost = np.array(self.r)
        self.demand_cost = np.array(self.k)
        # Last stage has 0 holding cost in original logic
        self.holding_cost = np.append(self.h, 0)
        self.supply_capacity = np.array(list(self.c))
        self.lead_time = np.array(list(self.L))
        self.discount = self.alpha

        # Retailer, Dist, Manuf, Supplier (Infinite)
        self.num_stages = len(self.init_inv) + 1
        m = self.num_stages
        lt_max = self.lead_time.max()

        # --- Spaces ---
        # Action space: Orders for Stages 0 to M-2 (Supplier M-1 is infinite)
        # Shape: (M-1,)
        # Note: or-gym uses int16, we use float32 for RL agent compatibility and cast inside step
        self.action_space = spaces.Box(
            low=np.zeros(m-1),
            high=self.supply_capacity.astype(np.float32),
            dtype=np.float32
        )

        # Observation space:
        # or-gym logic: Flattened array of [Inventory (M-1) + Action History (M-1 * lt_max)]
        # Total dim = (m-1) + (m-1) * lt_max = (m-1) * (1 + lt_max)
        self.pipeline_length = (m-1) * (lt_max + 1)

        # Note: or-gym defines low/high bounds loosely. We will follow suit.
        self.observation_space = spaces.Box(
            # Lost sales -> no negative inventory
            low=np.zeros(self.pipeline_length),
            high=np.ones(self.pipeline_length) *
            self.supply_capacity.max() * self.num_periods,
            dtype=np.float32  # Using float32 for torch compatibility
        )

        # Initialize internal state containers
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

        # 1. Inventory
        if t == 0:
            state[:m] = self.I0
        else:
            state[:m] = self.I[t]

        # 2. Action History (Pipeline)
        # The logic in or-gym is a bit implicit. It flattens the action log.
        # It takes the last `lt_max` actions.
        if t == 0:
            pass
        elif t >= lt_max:
            # Take actions from t-lt_max to t
            state[-m*lt_max:] += self.action_log[t-lt_max:t].flatten()
        else:
            # Take actions from 0 to t
            # The destination index needs to be carefully aligned to the end
            # or-gym: state[-m*(t):] += self.action_log[:t].flatten()
            # This fills from the RIGHT side of the buffer.
            state[-m*t:] += self.action_log[:t].flatten()

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        periods = self.num_periods
        m = self.num_stages

        # Initialize simulation trackers
        self.I = np.zeros([periods + 1, m - 1])  # Inventory
        self.T = np.zeros([periods + 1, m - 1])  # Pipeline
        self.R = np.zeros([periods, m - 1])     # Replenishment orders
        self.D = np.zeros(periods)              # Demand
        self.S = np.zeros([periods, m])         # Sales
        self.B = np.zeros([periods, m])         # Backlog (Unfilled)
        self.LS = np.zeros([periods, m])        # Lost Sales
        self.P = np.zeros(periods)              # Profit

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
        # Sanitize action
        action = np.array(action).reshape(-1)
        R = np.maximum(action, 0).astype(int)  # Order quantity request

        n = self.period
        L = self.lead_time
        I = self.I[n, :].copy()
        T = self.T[n, :].copy()
        m = self.num_stages
        c = self.supply_capacity

        # Log action
        self.action_log[n] = R.copy()

        # Available inventory at upstream (supplier) stages
        # Im1[i] is the inventory available for stage i to order from (which is Inventory at i+1)
        # Note: Inventory vector I is length m-1 (stages 0,1,2).
        # Im1 needs to represent inventory at 1, 2, and Infinity (for stage 3).
        Im1 = np.append(I[1:], np.inf)

        # --- Logic Step 1: Place Replenishment Order ---
        # Note: In Lost Sales case, we do NOT add backlog to R.
        # R = R (from agent)

        # Constrain R by Capacity and Upstream Inventory
        Rcopy = R.copy()  # Original request

        # Capacity constraint
        mask_cap = R >= c
        R[mask_cap] = c[mask_cap]

        # Upstream Inventory Constraint
        mask_inv = R >= Im1
        R[mask_inv] = Im1[mask_inv]

        self.R[n, :] = R  # Commit actual order

        # --- Logic Step 2: Receive Shipments (Pipeline arrival) ---
        RnL = np.zeros(m-1)
        for i in range(m-1):
            if n - L[i] >= 0:
                # Item ordered L[i] days ago arrives now
                RnL[i] = self.R[n - L[i], i].copy()
                I[i] = I[i] + RnL[i]

        # --- Logic Step 3: Demand Realization ---
        # Poisson distribution default
        D0 = np.random.poisson(self.dist_param['mu'])
        self.D[n] = D0
        D = D0  # Current demand

        # In Lost Sales, we do NOT add previous backlog to demand. D remains D0.

        # --- Logic Step 4: Sales ---
        # Retailer sales
        S0 = min(I[0], D)
        # Sales at upstream stages = Replenishment orders successfully filled by them
        S = np.append(S0, R)
        self.S[n, :] = S

        # --- Logic Step 5: Update Inventory & Pipeline ---
        I = I - S[:-1]  # Reduce inventory by sales (outflow)
        T = T - RnL + R  # Update pipeline (Inflow - Arrival + New Order)

        self.I[n+1, :] = I
        self.T[n+1, :] = T

        # --- Logic Step 6: Unfulfilled Orders ---
        # U = Demand - Sales (Retailer) + (Original Order - Fulfilled Order) (Upstream)
        U = np.append(D, Rcopy) - S

        # Lost Sales Logic
        LS = U
        self.LS[n, :] = LS
        self.B[n, :] = 0  # No backlog tracking in v1

        # --- Logic Step 7: Profit/Reward ---
        p = self.unit_price
        r = self.unit_cost
        k = self.demand_cost
        h = self.holding_cost
        a = self.discount

        # II: Augmented inventory to include last stage (0 cost)
        II = np.append(I, 0)
        # RR: Augmented replenishment to include last stage production (which is S[-1])
        RR = np.append(R, S[-1])

        # Profit P = Revenue - (Procurement + Shortage + Holding)
        # Discounted by alpha^n
        profit = a**n * np.sum(p*S - (r*RR + k*U + h*II))
        self.P[n] = profit

        # --- Finalize ---
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
