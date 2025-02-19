import logging
from typing import Callable, Dict, List, Tuple

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from .feature_importance import get_feature_importance

logger = logging.getLogger(__name__)
# Only add handler if logger doesn't already have handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def iterative_parameter_search(
    parameter_generator: Callable[[List[str], List[str], int], List[Dict]],
    objective_function: Callable[[Dict], float],
    initial_parameters: List[Dict],
    n_best_known_params: int,
    n_new_candidate_params: int,
    outer_iterations: int = 5,
    inner_iterations: int = 20,
    maximize: bool = True,
    neutral_value: float = 0.0,
    display_function: Callable[[Dict[str, float], List[Dict]], None] = None,
) -> Tuple[AxClient, List[str], List[str], Dict[str, float]]:
    """
    Run an iterative parameter search that combines best known parameters with new candidates.

    Args:
        parameter_generator: Function that takes (current_params, tried_params, num_params)
            and returns a list of new parameter configurations in Ax format
        objective_function: Function that takes parameter dict and returns objective value
        initial_parameters: List of parameter configurations in Ax format
        n_best_known_params: Number of highest-importance parameters to keep in each iteration
        n_new_candidate_params: Number of new parameters to try in each iteration
        outer_iterations: Number of outer loop iterations
        inner_iterations: Number of trials per inner optimization loop
        maximize: Whether to maximize (True) or minimize (False) the objective
        neutral_value: Value to use when computing feature importance
        display_function: Function to display intermediate results

    Returns:
        Tuple of (final AxClient, list of current parameter names, list of tried parameter names,
        dictionary of best importance scores for each parameter)
    """
    current_parameters = initial_parameters
    tried_parameter_names = []
    best_importance_scores = {}  # Track best importance scores seen for each parameter

    for outer_iter in range(outer_iterations):
        logger.info(f"Starting outer iteration {outer_iter + 1}/{outer_iterations}")

        # Create new AxClient with current parameters
        ax_client = AxClient(verbose_logging=False)
        ax_client.create_experiment(
            parameters=current_parameters,
            objectives={"objective": ObjectiveProperties(minimize=not maximize)},
        )

        # Run inner optimization loop
        if display_function is not None:
            display_function(best_importance_scores, current_parameters)

        for inner_iter in range(inner_iterations):
            parameters, trial_index = ax_client.get_next_trial()
            result = objective_function(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)

        # Get feature importance scores
        importance_scores = get_feature_importance(
            ax_client, neutral_value=neutral_value, maximize=maximize
        )
        print(f"Importance scores: {importance_scores}")

        # Update best importance scores
        for param_name, score in importance_scores.items():
            if (
                param_name not in best_importance_scores
                or score > best_importance_scores[param_name]
            ):
                best_importance_scores[param_name] = score

        # Select best known parameters based on importance scores
        sorted_params = sorted(
            best_importance_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_param_names = [
            param_name for param_name, _ in sorted_params[:n_best_known_params]
        ]

        # Keep the parameter configurations for best parameters
        retained_parameters = [
            param for param in current_parameters if param["name"] in best_param_names
        ]

        # Update tried parameters list with all current parameters not in best_param_names
        current_param_names = [param["name"] for param in current_parameters]
        tried_parameter_names.extend(
            name for name in current_param_names if name not in best_param_names
        )

        # Get new parameters from generator
        new_param_configs = parameter_generator(
            best_param_names, tried_parameter_names, n_new_candidate_params
        )

        if not new_param_configs:
            logger.info("No new parameters available. Stopping outer loop.")
            break

        # Combine retained and new parameters for next iteration
        current_parameters = retained_parameters + new_param_configs
        logger.info(f"Retained parameters: {best_param_names}")
        logger.info(f"New parameters added: {[p['name'] for p in new_param_configs]}")

    return ax_client, best_param_names, tried_parameter_names, best_importance_scores
