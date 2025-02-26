class ContinuousActionTarget:
    """
    Targets a specific action dimension in a continuous action space for reinforcement learning models.
    The target returns the selected action value (or a combination of them) from the model's output.
    """

    def __init__(self, action_indices=None, reduction='sum'):
        """
        :param action_indices: List of action indices to focus on. If None, considers all actions.
        :param reduction: How to aggregate multiple action values. Options: 'sum', 'mean', 'max'.
        """
        self.action_indices = action_indices
        self.reduction = reduction

    def __call__(self, model_output):
        """
        Extracts and aggregates the relevant action values from the model output.
        :param model_output: Tensor of shape (batch_size, action_dim)
        """
        if self.action_indices is None:
            selected_actions = model_output  # Use all action outputs
        else:
            selected_actions = model_output[:, self.action_indices]  # Select specific actions

        # Apply reduction method
        if self.reduction == 'sum':
            return selected_actions.sum(dim=-1)
        elif self.reduction == 'mean':
            return selected_actions.mean(dim=-1)
        elif self.reduction == 'max':
            return selected_actions.max(dim=-1)[0]
        else:
            raise ValueError("Invalid reduction method. Choose from 'sum', 'mean', or 'max'.")
