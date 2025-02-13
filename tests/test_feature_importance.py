# Standard library imports
from unittest.mock import MagicMock, patch

# Third-party imports
import pandas as pd
import pytest
import torch
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase

# Local imports
from feature_importance import (
    get_model_optimum_in_original_space,
    get_model_optimum_in_transformed_space,
)


class TestFeatureImportance(BotorchTestCase):
    def setUp(self):
        super().setUp()
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

        # Check shape and bounds
        self.assertEqual(result.shape, torch.Size([self.dim]))
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))
        self.assertEqual(result[0].item(), 0.5)  # Check fixed feature

    def test_get_model_optimum_minimize(self):
        # Test minimization
        fixed_features = {1: 0.3}  # Fix second dimension to 0.3
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=False
        )

        # Check shape and bounds
        self.assertEqual(result.shape, torch.Size([self.dim]))
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))
        self.assertAlmostEqual(result[1].item(), 0.3, places=6)  # Check fixed feature

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
        self.assertEqual(result.shape, torch.Size([self.dim]))
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))

        # Check all fixed features
        self.assertAlmostEqual(result[0].item(), 0.5, places=6)
        self.assertAlmostEqual(result[2].item(), 0.3, places=6)
        self.assertAlmostEqual(result[4].item(), 0.7, places=6)

    def test_no_fixed_features(self):
        # Test optimization with no fixed features
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, {}, maximize=True
        )

        # Check shape and bounds
        self.assertEqual(result.shape, torch.Size([self.dim]))
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))

    def test_all_features_fixed(self):
        # Test with all features fixed
        fixed_features = {i: 0.5 for i in range(self.dim)}
        result = get_model_optimum_in_transformed_space(
            self.ax_model, self.bounds, fixed_features, maximize=True
        )

        # Check shape and bounds
        self.assertEqual(result.shape, torch.Size([self.dim]))

        # All values should be 0.5
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.5)))

    def test_get_model_optimum_in_original_space(self):
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
        with patch(
            "feature_importance.get_model_optimum_in_transformed_space"
        ) as mock_transform_opt:
            mock_transform_opt.return_value = torch.tensor([0.5, 0.7])

            result = get_model_optimum_in_original_space(
                mock_client, fixed_features, maximize=True
            )

            self.assertAlmostEqual(result["x1"], 5.0, places=4)
            self.assertEqual(set(result.keys()), {"x1", "x2"})
            self.assertTrue(1 <= result["x1"] <= 10)
            self.assertTrue(1 <= result["x2"] <= 100)

    def test_get_model_optimum_in_original_space_no_fixed_features(self):
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
        with patch(
            "feature_importance.get_model_optimum_in_transformed_space"
        ) as mock_transform_opt:
            mock_transform_opt.return_value = torch.tensor([0.3, 0.7])

            result = get_model_optimum_in_original_space(
                mock_client, {}, maximize=False
            )

            self.assertEqual(set(result.keys()), {"x1", "x2"})
            self.assertTrue(all(0 <= v <= 1 for v in result.values()))

    def test_get_model_optimum_in_original_space_invalid_feature(self):
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
        with self.assertRaises(ValueError):
            get_model_optimum_in_original_space(
                mock_client, {"invalid_feature": 0.5}, maximize=True
            )
