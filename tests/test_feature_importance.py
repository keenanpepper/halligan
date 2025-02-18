# Standard library imports
import subprocess
import sys
from unittest.mock import MagicMock

# Third-party imports
import pandas as pd
import pytest
import torch

# Local imports
from halligan.feature_importance import (
    get_model_optimum_in_original_space,
    get_model_optimum_in_transformed_space,
)


@pytest.fixture(params=["0.4.3", "0.5.0"], scope="session")
def ax_version(request):
    print(f"Running with Ax version {request.param}")

    # Store current installed version if any
    try:
        import ax
        import pkg_resources

        old_version = pkg_resources.get_distribution("ax-platform").version
    except (ImportError, pkg_resources.DistributionNotFound):
        old_version = None

    # Install requested version
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", f"ax-platform=={request.param}"]
    )

    yield request.param

    # Restore previous version if there was one
    if old_version:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"ax-platform=={old_version}"]
        )


@pytest.mark.usefixtures("ax_version")
class TestFeatureImportance:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)

        from ax.core.search_space import SearchSpaceDigest
        from ax.models.torch.botorch_modular.model import BoTorchModel
        from ax.models.torch.botorch_modular.surrogate import Surrogate
        from botorch.models import SingleTaskGP
        from botorch.test_functions import Hartmann
        from botorch.utils.datasets import SupervisedDataset

        # Create a simple test function (Hartmann6) and generate some training data
        self.dim = 6
        self.train_X = torch.rand(10, self.dim)
        self.test_fn = Hartmann(dim=self.dim)
        self.train_Y = self.test_fn(self.train_X).unsqueeze(-1)

        # Create feature names
        feature_names = [f"x{i}" for i in range(self.dim)]

        # Create dataset
        dataset = SupervisedDataset(
            X=self.train_X,
            Y=self.train_Y,
            feature_names=feature_names,
            outcome_names=["objective"],
        )

        # Create search space digest
        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names, bounds=[(0.0, 1.0) for _ in range(self.dim)]
        )

        # Create and fit surrogate
        surrogate = Surrogate(botorch_model_class=SingleTaskGP)
        surrogate.fit(datasets=[dataset], search_space_digest=search_space_digest)

        self.ax_model = BoTorchModel(surrogate=surrogate)

        # Define bounds
        self.bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])

    def test_get_model_optimum_maximize(self):
        # Test maximization
        fixed_features = {0: 0.5}  # Fix first dimension to 0.5
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=True
        )

        # Get predicted value at optimum
        pred_opt = self.ax_model.predict(result.unsqueeze(0))[0].item()

        # Get predicted value at random point (with same fixed feature)
        random_point = torch.rand(self.dim)
        random_point[0] = 0.5  # Match fixed feature
        pred_random = self.ax_model.predict(random_point.unsqueeze(0))[0].item()

        # Optimum should be better than random point
        assert pred_opt > pred_random, (
            "Optimum should give better prediction than random point"
        )

        # Original assertions
        assert result.shape == torch.Size([self.dim])
        assert torch.all(result >= 0) and torch.all(result <= 1)
        assert result[0].item() == pytest.approx(0.5)  # Check fixed feature

    def test_get_model_optimum_minimize(self):
        # Test minimization
        fixed_features = {1: 0.3}  # Fix second dimension to 0.3
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=False
        )

        # Check shape and bounds
        assert result.shape == torch.Size([self.dim])
        assert torch.all(result >= 0) and torch.all(result <= 1)
        assert result[1].item() == pytest.approx(0.3)  # Check fixed feature

    def test_invalid_fixed_features(self):
        # Test with invalid fixed feature index
        fixed_features = {self.dim + 1: 0.5}  # Invalid dimension
        with pytest.raises(IndexError):
            get_model_optimum_in_transformed_space(
                self.ax_model, self.bounds, fixed_features, maximize=True
            )

    def test_fixed_features_out_of_bounds(self):
        # Test with fixed feature value outside bounds
        fixed_features = {0: 1.5}  # Outside [0,1] bounds
        with pytest.raises(ValueError):
            get_model_optimum_in_transformed_space(
                self.ax_model, self.bounds, fixed_features, maximize=True
            )

    def test_multiple_fixed_features(self):
        # Test with multiple fixed features
        fixed_features = {0: 0.5, 2: 0.3, 4: 0.7}
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=True
        )

        # Check shape and bounds
        assert result.shape == torch.Size([self.dim])
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check all fixed features
        assert result[0].item() == pytest.approx(0.5)
        assert result[2].item() == pytest.approx(0.3)
        assert result[4].item() == pytest.approx(0.7)

    def test_no_fixed_features(self):
        # Test optimization with no fixed features
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, {}, maximize=True
        )

        # Check shape and bounds
        assert result.shape == torch.Size([self.dim])
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_all_features_fixed(self):
        # Test with all features fixed
        fixed_features = {i: 0.5 for i in range(self.dim)}
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=True
        )

        # Check shape and bounds
        assert result.shape == torch.Size([self.dim])

        # All values should be 0.5
        assert torch.allclose(result, torch.full_like(result, 0.5))

    def test_get_model_optimum_in_original_space(self):
        from ax.core.arm import Arm
        from ax.core.data import Data
        from ax.core.experiment import Experiment
        from ax.core.metric import Metric
        from ax.core.objective import Objective
        from ax.core.observation import ObservationFeatures
        from ax.core.optimization_config import OptimizationConfig
        from ax.core.parameter import ParameterType, RangeParameter
        from ax.core.search_space import SearchSpace
        from ax.modelbridge.torch import TorchModelBridge

        # Create a mock Ax client with a simple search space
        mock_client = MagicMock()

        # Create a search space with both linear parameters
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=1, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=1, upper=100
            ),
        ]
        search_space = SearchSpace(parameters=parameters)

        # Create experiment with optimization config
        experiment = Experiment(
            search_space=search_space,
            name="test_experiment",
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="objective"), minimize=False)
            ),
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 5.0, "x2": 50.0}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

        # Create data using pandas DataFrame
        df = pd.DataFrame(
            {
                "arm_name": ["0_0"],
                "metric_name": ["objective"],
                "mean": [1.0],
                "sem": [0.0],
                "trial_index": [0],
            }
        )
        data = Data(df)

        # Set up the mock experiment and model bridge
        mock_client.experiment.search_space = search_space
        model_bridge = TorchModelBridge(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=self.ax_model,
            transforms=[],
        )
        mock_client.generation_strategy.model = model_bridge

        # Test maximization with fixed features
        fixed_features = {"x1": 5.0}
        result = get_model_optimum_in_original_space(
            mock_client, fixed_features, maximize=True
        )

        # Verify the results
        assert result["x1"] == pytest.approx(
            5.0, rel=1e-4
        )  # Fixed feature should be exactly 5.0
        assert set(result.keys()) == {"x1", "x2"}
        assert 1 <= result["x1"] <= 10
        assert 1 <= result["x2"] <= 100

        # Test that the result actually improves the objective
        # Create observation features from the optimized parameters
        test_features = [ObservationFeatures(parameters=result)]
        prediction = model_bridge.predict(test_features)[0]["objective"][
            0
        ]  # Get prediction for 'objective' metric

        # Create random observation for comparison
        random_params = {"x1": 5.0, "x2": 50.0}  # Keep x1 fixed
        random_features = [ObservationFeatures(parameters=random_params)]
        random_prediction = model_bridge.predict(random_features)[0]["objective"][0]

        # The optimized result should be at least as good as the random point
        # (within numerical precision)
        assert prediction == pytest.approx(
            random_prediction, rel=1e-4
        )  # Allow for small relative differences

    def test_get_model_optimum_in_original_space_no_fixed_features(self):
        from ax.core.arm import Arm
        from ax.core.data import Data
        from ax.core.experiment import Experiment
        from ax.core.metric import Metric
        from ax.core.objective import Objective
        from ax.core.observation import ObservationFeatures
        from ax.core.optimization_config import OptimizationConfig
        from ax.core.parameter import ParameterType, RangeParameter
        from ax.core.search_space import SearchSpace
        from ax.modelbridge.torch import TorchModelBridge

        # Create a mock Ax client with a simple search space
        mock_client = MagicMock()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=1
            ),
        ]
        search_space = SearchSpace(parameters=parameters)

        # Create experiment with optimization config
        experiment = Experiment(
            search_space=search_space,
            name="test_experiment",
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="objective"), minimize=False)
            ),
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 0.5, "x2": 0.5}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

        # Create data using pandas DataFrame
        df = pd.DataFrame(
            {
                "arm_name": ["0_0"],
                "metric_name": ["objective"],
                "mean": [1.0],
                "sem": [0.0],
                "trial_index": [0],
            }
        )
        data = Data(df)

        mock_client.experiment.search_space = search_space
        model_bridge = TorchModelBridge(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=self.ax_model,
            transforms=[],
        )
        mock_client.generation_strategy.model = model_bridge

        # Test with no fixed features
        result = get_model_optimum_in_original_space(mock_client, {}, maximize=False)

        # Basic checks on the result
        assert set(result.keys()) == {"x1", "x2"}
        assert all(0 <= v <= 1 for v in result.values())

        # Test that the result actually improves the objective
        test_features = [ObservationFeatures(parameters=result)]
        prediction = model_bridge.predict(test_features)[0]["objective"][0]

        # Create random observation for comparison
        random_params = {"x1": 0.5, "x2": 0.5}
        random_features = [ObservationFeatures(parameters=random_params)]
        random_prediction = model_bridge.predict(random_features)[0]["objective"][0]

        # Since maximize=False, the prediction should be less than or equal to random
        # (within numerical precision)
        assert (
            prediction <= random_prediction + 1e-6
        )  # Allow for small numerical differences

    def test_get_model_optimum_in_original_space_invalid_feature(self):
        from ax.core.arm import Arm
        from ax.core.data import Data
        from ax.core.experiment import Experiment
        from ax.core.metric import Metric
        from ax.core.objective import Objective
        from ax.core.optimization_config import OptimizationConfig
        from ax.core.parameter import ParameterType, RangeParameter
        from ax.core.search_space import SearchSpace
        from ax.modelbridge.torch import TorchModelBridge

        # Create a mock Ax client
        mock_client = MagicMock()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=1
            )
        ]
        search_space = SearchSpace(parameters=parameters)

        # Create experiment with optimization config
        experiment = Experiment(
            search_space=search_space,
            name="test_experiment",
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="objective"), minimize=False)
            ),
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(parameters={"x1": 0.5}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

        # Create data using pandas DataFrame
        df = pd.DataFrame(
            {
                "arm_name": ["0_0"],
                "metric_name": ["objective"],
                "mean": [1.0],
                "sem": [0.0],
                "trial_index": [0],
            }
        )
        data = Data(df)

        mock_client.experiment.search_space = search_space
        model_bridge = TorchModelBridge(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=self.ax_model,
            transforms=[],
        )
        mock_client.generation_strategy.model = model_bridge

        # Test with invalid feature name
        with pytest.raises(ValueError, match="'invalid_feature' is not in list"):
            get_model_optimum_in_original_space(
                mock_client, {"invalid_feature": 0.5}, maximize=True
            )

        # Also test with out-of-bounds value for valid feature
        with pytest.raises(
            ValueError,
            match="Fixed feature 0 with value 2.0 is outside bounds \[0.0, 1.0\]",
        ):
            get_model_optimum_in_original_space(mock_client, {"x1": 2.0}, maximize=True)
