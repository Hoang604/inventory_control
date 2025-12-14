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

    Why this is 'smarter' than Base-Stock:
    - Base-Stock orders every single period to refill exactly what was sold.
    - (s, S) creates a buffer. It only orders when inventory is truly low (below s).
    - This reduces 'ordering nervousness' and can handle lumpiness in demand or 
      setup costs better than simple Base-Stock.
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
