import numpy as np


class RolloutBuffer:
    """
    A class used to store the trajectories of an environment/agent
    pair for PPO training.
    """

    def __init__(
        self,
        total_buffer_size: int,
        observation_dim: int,
        observation_length: int | list | tuple,
        action_dim: int,
    ) -> None:
        """
        Initializes the RolloutBuffer object.

        Parameters
        ----------
        total_buffer_size : int
            The total size of the buffer.
        observation_dim : int
            The dimension of the observation space.
        action_dim : int
            The dimension of the action space.
        """
        # Store the input parameters
        self.total_buffer_size = total_buffer_size
        self.observation_dim = observation_dim
        self.observation_length = observation_length
        self.action_dim = action_dim

        # Initialize buffers with provided dimensions
        if isinstance(observation_length, (tuple, list)):
            obs_dim = (total_buffer_size, *observation_length, observation_dim)
        else:
            obs_dim = (total_buffer_size, observation_length, observation_dim)
        self.observation_buffer = np.zeros(obs_dim, dtype=np.float32)
        self.termination_buffer = np.zeros(total_buffer_size, dtype=bool)
        self.action_buffer = np.zeros((total_buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(total_buffer_size, dtype=np.float32)
        self.value_buffer = np.zeros(total_buffer_size, dtype=np.float32)
        self.log_prob_buffer = np.zeros(total_buffer_size, dtype=np.float32)

        # Track the actual number of valid entries in the buffer
        self.buffer_size = 0

    def reset(self):
        """
        Resets the buffers to their initial state.
        """
        self.observation_buffer.fill(0)
        self.termination_buffer.fill(0)
        self.action_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.value_buffer.fill(0)
        self.log_prob_buffer.fill(0)
        self.buffer_size = 0

    @property
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return self.buffer_size == self.total_buffer_size

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

        # Store the current buffer index
        buffer_idx = self.buffer_size
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

            # Clamp to buffer size
            end_idx = min(end_idx, self.total_buffer_size - buffer_idx)

            # Store the valid steps for this batch element
            if end_idx > 0:
                self.observation_buffer[buffer_idx : buffer_idx + end_idx] = (
                    observations[batch_idx, :end_idx]
                )
                self.action_buffer[buffer_idx : buffer_idx + end_idx] = actions[
                    batch_idx, :end_idx
                ]
                self.reward_buffer[buffer_idx : buffer_idx + end_idx] = rewards[
                    batch_idx, :end_idx
                ]
                self.value_buffer[buffer_idx : buffer_idx + end_idx] = values[
                    batch_idx, :end_idx
                ]
                self.log_prob_buffer[buffer_idx : buffer_idx + end_idx] = log_probs[
                    batch_idx, :end_idx
                ]
                self.termination_buffer[buffer_idx : buffer_idx + end_idx] = (
                    terminations[batch_idx, :end_idx]
                )
                # Update the buffer index
                buffer_idx += end_idx

        # Update the actual buffer size
        self.buffer_size = buffer_idx

    def get_buffer_data(self):
        """
        Returns the valid data from the buffer.

        Returns
        -------
        dict
            Dictionary containing all buffer data up to buffer_size.
        """
        return {
            "observations": self.observation_buffer[: self.buffer_size],
            "actions": self.action_buffer[: self.buffer_size],
            "rewards": self.reward_buffer[: self.buffer_size],
            "values": self.value_buffer[: self.buffer_size],
            "log_probs": self.log_prob_buffer[: self.buffer_size],
            "terminations": self.termination_buffer[: self.buffer_size],
        }
