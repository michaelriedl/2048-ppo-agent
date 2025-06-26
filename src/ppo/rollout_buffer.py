import numpy as np


class RolloutBuffer:
    """
    A class used to store the trajectories of an environment/agent
    pair for PPO training.
    """

    def __init__(
        self,
        batch_size: int,
        obs_dim: int,
        act_dim: int,
        max_steps: int,
        gamma=0.99,
        lam=0.95,
    ) -> None:
        """
        Initializes the RolloutBuffer object.

        Parameters
        ----------
        batch_size : int
            The size of the batch input to the buffer.
        obs_dim : int
            The dimension of the observation space.
        act_dim : int
            The dimension of the action space.
        max_steps : int
            The maximum number of steps to store in the buffer.
        gamma : float, optional
            The discount factor, by default 0.99.
        lam : float, optional
            The GAE lambda parameter, by default 0.95.
        """
        self.obs_buffer = np.zeros((batch_size, max_steps, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros((batch_size, max_steps, act_dim), dtype=np.float32)
        self.adv_buffer = np.zeros((batch_size, max_steps), dtype=np.float32)
        self.rew_buffer = np.zeros((batch_size, max_steps), dtype=np.float32)
        self.ret_buffer = np.zeros((batch_size, max_steps), dtype=np.float32)
        self.val_buffer = np.zeros((batch_size, max_steps), dtype=np.float32)
        self.logp_buffer = np.zeros((batch_size, max_steps), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
