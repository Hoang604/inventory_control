import numpy as np


class BaseStockPolicy:
    """
    Implements a Base-Stock Policy (Order-Up-To Policy).

    For a serial supply chain, this typically operates on 'Echelon Inventory Position'.
    Echelon Inventory Position at stage i = 
        (Inventory at i + Pipeline at i) + (Inventory at i-1 + Pipeline at i-1) + ... + (Inventory at 0 + Pipeline at 0)

    Policy:
        Order_i = Target_i - Echelon_IP_i

    This heuristic tries to keep the total stock in the subsystem (stage i and all downstream) 
    at a specific target level 'z'.
    """

    def __init__(self, env, z=None):
        """
        Args:
            env: The environment instance (needed to access internal state like pipeline).
            z (list or np.array): The base-stock levels for each stage. 
                                  If None, must be provided when calling get_action.
        """
        self.env = env
        self.z = np.array(z) if z is not None else None

    def get_action(self, z=None):
        """
        Calculates the order quantities based on the current state and base-stock levels.

        Args:
            z (np.array, optional): Override the stored base-stock levels.

        Returns:
            np.array: Action vector (order quantities).
        """
        if z is None:
            z = self.z
        if z is None:
            raise ValueError("Base stock levels 'z' must be provided.")

        period = self.env.period


        if period >= len(self.env.I):
            current_I = self.env.I[-1]
            current_T = self.env.T[-1]
        else:
            current_I = self.env.I[period]
            current_T = self.env.T[period]



        ip_local = current_I + current_T
        ip_echelon = np.cumsum(ip_local)

        reorder = z - ip_echelon

        action = np.maximum(reorder, 0)

        return action.astype(np.float32)


class MinMaxPolicy:
    """
    Implements an (s, S) Order Policy (Min-Max Policy).

    Logic for each stage i (using Echelon Inventory Position):
        If Echelon_IP_i < s_i (Reorder Point):
            Order_i = S_i - Echelon_IP_i (Order up to Target S_i)
        Else:
            Order_i = 0
    """

    def __init__(self, env, min_max_params=None):
        """
        Args:
            env: The environment instance.
            min_max_params (list of tuples/arrays): pairs of (s, S) for each stage.
                                                    Format: [(s0, S0), (s1, S1), ...]
        """
        self.env = env
        self.params = np.array(
            min_max_params) if min_max_params is not None else None

    def get_action(self, params=None):
        """
        Calculates (s, S) order quantities.
        """
        if params is None:
            params = self.params
        if params is None:
            params = np.array([[0, 100], [0, 100], [0, 100]])

        params = np.array(params)

        s_levels = params[:, 0]
        S_levels = params[:, 1]

        period = self.env.period

        if period >= len(self.env.I):
            current_I = self.env.I[-1]
            current_T = self.env.T[-1]
        else:
            current_I = self.env.I[period]
            current_T = self.env.T[period]

        ip_local = current_I + current_T
        ip_echelon = np.cumsum(ip_local)

        action = np.zeros_like(ip_echelon)

        reorder_mask = ip_echelon < s_levels

        if np.any(reorder_mask):
            action[reorder_mask] = S_levels[reorder_mask] - \
                ip_echelon[reorder_mask]

        action = np.maximum(action, 0)

        return action.astype(np.float32)


class RQPolicy:
    """
    Implements an (R, Q) Fixed Order Quantity Policy.

    Logic for each stage i (using Echelon Inventory Position):
        If Echelon_IP_i < R_i (Reorder Point):
            Order_i = Q_i (Fixed order quantity)
        Else:
            Order_i = 0

    Unlike (s,S) which orders up-to S, this orders a fixed quantity Q.
    """

    def __init__(self, env, rq_params=None):
        """
        Args:
            env: The environment instance.
            rq_params (list of tuples/arrays): pairs of (R, Q) for each stage.
                                               Format: [(R0, Q0), (R1, Q1), ...]
        """
        self.env = env
        self.params = np.array(rq_params) if rq_params is not None else None

    def get_action(self, params=None):
        """
        Calculates (R, Q) order quantities.
        """
        if params is None:
            params = self.params
        if params is None:
            # Default: reorder at 50, order 30 units
            params = np.array([[50, 30], [50, 30], [50, 30]])

        params = np.array(params)
        R_levels = params[:, 0]  # Reorder points
        Q_levels = params[:, 1]  # Fixed order quantities

        period = self.env.period

        if period >= len(self.env.I):
            current_I = self.env.I[-1]
            current_T = self.env.T[-1]
        else:
            current_I = self.env.I[period]
            current_T = self.env.T[period]

        ip_local = current_I + current_T
        ip_echelon = np.cumsum(ip_local)

        action = np.zeros_like(ip_echelon)

        # Order fixed Q when below reorder point R
        reorder_mask = ip_echelon < R_levels
        action[reorder_mask] = Q_levels[reorder_mask]

        action = np.maximum(action, 0)

        return action.astype(np.float32)


class PeriodicReviewPolicy:
    """
    Implements a (T, S) Periodic Review Policy.

    Logic:
        Every T periods, review inventory and order up to S.
        On non-review periods, order nothing.

    This simulates businesses that only place orders on specific days
    (e.g., weekly ordering, monthly ordering).
    """

    def __init__(self, env, review_period=5, S_levels=None):
        """
        Args:
            env: The environment instance.
            review_period: Number of periods between reviews (T).
            S_levels: Order-up-to levels for each stage.
        """
        self.env = env
        self.review_period = review_period
        self.S_levels = np.array(S_levels) if S_levels is not None else None

    def get_action(self, review_period=None, S_levels=None):
        """
        Calculates periodic review order quantities.
        """
        if review_period is None:
            review_period = self.review_period
        if S_levels is None:
            S_levels = self.S_levels
        if S_levels is None:
            S_levels = np.array([150, 200, 300])

        period = self.env.period

        # Only order on review periods
        if period % review_period != 0:
            num_stages = len(self.env.I[0]) if len(self.env.I) > 0 else 3
            return np.zeros(num_stages, dtype=np.float32)

        if period >= len(self.env.I):
            current_I = self.env.I[-1]
            current_T = self.env.T[-1]
        else:
            current_I = self.env.I[period]
            current_T = self.env.T[period]

        ip_local = current_I + current_T
        ip_echelon = np.cumsum(ip_local)

        # Order up to S on review periods
        action = S_levels - ip_echelon
        action = np.maximum(action, 0)

        return action.astype(np.float32)


class LotForLotPolicy:
    """
    Implements a Lot-for-Lot (L4L) Policy.

    Logic:
        Order exactly the expected demand for next period.
        Minimal inventory holding, just-in-time approach.

    For multi-echelon:
        Stage 0 orders expected demand (mu)
        Upstream stages order what downstream ordered
    """

    def __init__(self, env, expected_demand=None):
        """
        Args:
            env: The environment instance.
            expected_demand: Expected demand per period (defaults to env's mu).
        """
        self.env = env
        self.expected_demand = expected_demand

    def get_action(self, expected_demand=None):
        """
        Calculates lot-for-lot order quantities.
        """
        if expected_demand is None:
            expected_demand = self.expected_demand
        if expected_demand is None:
            expected_demand = self.env.dist_param.get('mu', 20)

        period = self.env.period
        num_stages = len(self.env.I[0]) if len(self.env.I) > 0 else 3

        # Stage 0: order expected demand
        # Upstream stages: order what was demanded from them (proxy: expected demand)
        # In practice, this creates a "pull" system
        action = np.full(num_stages, expected_demand, dtype=np.float32)

        # Adjust based on current inventory - don't over-order if we have stock
        if period >= len(self.env.I):
            current_I = self.env.I[-1]
        else:
            current_I = self.env.I[period]

        # Reduce order if we have excess inventory
        for i in range(num_stages):
            if current_I[i] > expected_demand:
                action[i] = max(0, expected_demand - (current_I[i] - expected_demand))

        return action.astype(np.float32)


class NoisyBaseStockPolicy:
    """
    Implements a Noisy Base-Stock Policy for exploration.

    Logic:
        Base order = BaseStockPolicy order
        Final order = Base order + Gaussian noise

    This creates diverse sub-optimal trajectories useful for offline RL.
    """

    def __init__(self, env, z=None, noise_std=10.0):
        """
        Args:
            env: The environment instance.
            z: Base-stock levels for each stage.
            noise_std: Standard deviation of Gaussian noise to add.
        """
        self.env = env
        self.z = np.array(z) if z is not None else None
        self.noise_std = noise_std
        self.base_policy = BaseStockPolicy(env, z)

    def get_action(self, z=None, noise_std=None):
        """
        Calculates noisy base-stock order quantities.
        """
        if z is None:
            z = self.z
        if z is None:
            z = np.array([100, 150, 200])
        if noise_std is None:
            noise_std = self.noise_std

        # Get base-stock action
        self.base_policy.z = z
        base_action = self.base_policy.get_action()

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, size=base_action.shape)
        noisy_action = base_action + noise

        # Clip to valid range
        noisy_action = np.maximum(noisy_action, 0)

        return noisy_action.astype(np.float32)
