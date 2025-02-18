import torch
from ax.core.observation import ObservationFeatures
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from botorch.acquisition import PosteriorMean
from botorch.optim.optimize import optimize_acqf


def get_model_optimum_in_transformed_space(
    ax_model, bounds, fixed_features, maximize=True
):
    """
    Find the optimum of a model given bounds and fixed features.

    Everything is done in transformed space (the space of the surrogate model).

    Args:
        ax_model: BoTorchModel instance
        bounds: tensor of shape (2, dim) containing lower and upper bounds
        fixed_features: dict of {feature_index: value} for fixed dimensions
        maximize: bool, whether to maximize or minimize

    Returns:
        tensor of shape (dim,) containing the optimal point

    Raises:
        ValueError: If any fixed feature value is outside its bounds
    """
    # Check if fixed features are within bounds
    for idx, value in fixed_features.items():
        if value < bounds[0][idx] or value > bounds[1][idx]:
            raise ValueError(
                f"Fixed feature {idx} with value {value} is outside bounds [{bounds[0][idx]}, {bounds[1][idx]}]"
            )

    # Get the underlying BoTorch model through the surrogate
    botorch_model = ax_model.surrogate.model

    acq_function = PosteriorMean(botorch_model, maximize=maximize)

    optimal_point, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
        fixed_features=fixed_features,
        return_best_only=True,
    )

    return optimal_point.squeeze(0)


def get_model_optimum_in_original_space(ax_client, fixed_features, maximize=True):
    """
    Find the optimum of a model given a search space and fixed features.

    Inputs and outputs are in the original space.

    Args:
        ax_client: AxClient instance (for the ModelBridge and SearchSpace)
        fixed_features: dict of {feature_name: value} for fixed features
        maximize: bool, whether to maximize or minimize

    Returns:
        dict of {feature_name: value} for the optimal point
    """
    # Transform search space and extract bounds
    model_bridge = ax_client.generation_strategy.model
    search_space = ax_client.experiment.search_space.clone()
    for transform in model_bridge.transforms.values():
        search_space = transform.transform_search_space(search_space)

    search_space_digest = extract_search_space_digest(
        search_space, list(search_space.parameters.keys())
    )

    bounds = torch.tensor(search_space_digest.bounds).transpose(0, 1)

    # Transform fixed features
    fixed_features_transformed = [
        ObservationFeatures(
            parameters={
                feature_name: value for feature_name, value in fixed_features.items()
            }
        )
    ]
    for transform in model_bridge.transforms.values():
        fixed_features_transformed = transform.transform_observation_features(
            fixed_features_transformed
        )

    # Convert feature names to indices in fixed_features_transformed
    fixed_features_by_index = {}
    feature_names = list(search_space.parameters.keys())
    for feature_name, value in fixed_features_transformed[0].parameters.items():
        feature_idx = feature_names.index(feature_name)
        fixed_features_by_index[feature_idx] = value

    # Call actual optimizer
    result_in_transformed_space = get_model_optimum_in_transformed_space(
        model_bridge.model, bounds, fixed_features_by_index, maximize
    )

    # Convert result to original space
    result_in_transformed_space_by_name = [
        ObservationFeatures(
            parameters={
                feature_names[i]: result_in_transformed_space[i]
                for i in range(len(feature_names))
            }
        )
    ]
    result_in_original_space = result_in_transformed_space_by_name
    for transform in model_bridge.transforms.values():
        result_in_original_space = transform.untransform_observation_features(
            result_in_original_space
        )

    return result_in_original_space[0].parameters
