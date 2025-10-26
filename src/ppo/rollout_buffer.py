import numpy as np


class RolloutBuffer:
    """
    A class used to store the trajectories of an environment/agent
    pair for PPO training.
    """

    def __init__(
        self,
        observation_dim: int,
        observation_length: int | list | tuple,
        action_dim: int,
    ) -> None:
        """
        Initializes the RolloutBuffer object.

        Parameters
        ----------
        observation_dim : int
            The dimension of the observation space.
        observation_length : int | list | tuple
            The length/shape of the observation sequence.
        action_dim : int
            The dimension of the action space.
        """
        # Store the input parameters
        self.observation_dim = observation_dim
        self.observation_length = observation_length
        self.action_dim = action_dim

        # Initialize buffers as lists
        self.observation_buffer = []
        self.termination_buffer = []
        self.action_buffer = []
        self.action_mask_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []

        # Track the actual number of valid entries in the buffer
        self.buffer_size = 0

    def reset(self):
        """
        Resets the buffers to their initial state.
        """
        self.observation_buffer = []
        self.termination_buffer = []
        self.action_buffer = []
        self.action_mask_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.log_prob_buffer = []
        self.buffer_size = 0

    def _validate_and_reshape_observations(
        self, observations: np.ndarray
    ) -> np.ndarray:
        """
        Validates and reshapes observations to match the expected buffer shape.

        Parameters
        ----------
        observations : np.ndarray
            The input observations array. Expected shape is
            (batch_size, time_steps, *observation_length, observation_dim)

        Returns
        -------
        np.ndarray
            Validated and potentially reshaped observations

        Raises
        ------
        ValueError
            If observations cannot be reshaped to match expected dimensions
        """
        # Expected shape for each timestep: (*observation_length, observation_dim)
        if isinstance(self.observation_length, (tuple, list)):
            expected_obs_dims = (*self.observation_length, self.observation_dim)
        else:
            expected_obs_dims = (self.observation_length, self.observation_dim)

        # Validate input dimensions
        if len(observations.shape) < 2:
            raise ValueError(
                f"Observations must have at least 2 dimensions (batch_size, time_steps, ...), "
                f"but got shape {observations.shape}"
            )

        batch_size, time_steps = observations.shape[:2]

        # Check if observations already match expected shape
        if observations.shape[2:] == expected_obs_dims:
            return observations

        # Try to reshape observations to match expected shape
        try:
            # Calculate expected total elements per timestep
            expected_obs_elements = np.prod(expected_obs_dims)

            # Flatten the observation dimensions after batch_size and time_steps
            flattened_obs = observations.reshape(batch_size, time_steps, -1)

            # Check if flattened observations can be reshaped to expected shape
            if flattened_obs.shape[2] == expected_obs_elements:
                # Reshape to expected shape
                reshaped_obs = flattened_obs.reshape(
                    batch_size, time_steps, *expected_obs_dims
                )
                return reshaped_obs
            else:
                raise ValueError(
                    f"Cannot reshape observations from shape {observations.shape} to expected shape "
                    f"(batch_size, time_steps, {expected_obs_dims}). "
                    f"Flattened observations have {flattened_obs.shape[2]} elements per timestep "
                    f"but expected {expected_obs_elements} elements."
                )

        except Exception as e:
            raise ValueError(
                f"Failed to reshape observations from shape {observations.shape} to expected shape "
                f"(batch_size, time_steps, {expected_obs_dims}). Error: {str(e)}"
            )

    def store_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        terminations: np.ndarray,
    ):
        """
        Stores a batch of data in the buffer.

        Parameters
        ----------
        observations : np.ndarray
            The observations for the batch. Shape: (batch_size, time_steps, observation_dim)
        actions : np.ndarray
            The actions taken in the batch. Shape: (batch_size, time_steps, action_dim)
        action_masks : np.ndarray
            The action masks for the batch. Shape: (batch_size, time_steps, action_dim)
        rewards : np.ndarray
            The rewards received in the batch. Shape: (batch_size, time_steps)
        values : np.ndarray
            The value estimates for the batch. Shape: (batch_size, time_steps)
        log_probs : np.ndarray
            The log probabilities of the actions taken. Shape: (batch_size, time_steps)
        terminations : np.ndarray
            The termination flags for the batch. Shape: (batch_size, time_steps)
        """
        # Validate and reshape observations if necessary
        observations = self._validate_and_reshape_observations(observations)

        # Get batch size from the observations array
        batch_size = observations.shape[0]
        # Iterate through each batch element
        for batch_idx in range(batch_size):
            # Find the first termination index for this batch element
            termination_indices = np.where(terminations[batch_idx])[0]
            if len(termination_indices) > 0:
                # Include the termination step
                end_idx = termination_indices[0] + 1
            else:
                # If no termination found, don't include any steps
                end_idx = 0

            # Store the valid steps for this batch element
            if end_idx > 0:
                # Append data to lists
                for i in range(end_idx):
                    self.observation_buffer.append(observations[batch_idx, i])
                    self.action_buffer.append(actions[batch_idx, i])
                    self.action_mask_buffer.append(action_masks[batch_idx, i])
                    self.reward_buffer.append(rewards[batch_idx, i])
                    self.value_buffer.append(values[batch_idx, i])
                    self.log_prob_buffer.append(log_probs[batch_idx, i])
                    self.termination_buffer.append(terminations[batch_idx, i])
                
                # Update the buffer size
                self.buffer_size += end_idx

    def get_buffer_data(self):
        """
        Returns the valid data from the buffer.

        Returns
        -------
        dict
            Dictionary containing all buffer data as numpy arrays.
        """
        return {
            "observations": np.array(self.observation_buffer, dtype=np.float32),
            "actions": np.array(self.action_buffer, dtype=np.float32),
            "action_masks": np.array(self.action_mask_buffer, dtype=bool),
            "rewards": np.array(self.reward_buffer, dtype=np.float32),
            "values": np.array(self.value_buffer, dtype=np.float32),
            "log_probs": np.array(self.log_prob_buffer, dtype=np.float32),
            "terminations": np.array(self.termination_buffer, dtype=bool),
        }
