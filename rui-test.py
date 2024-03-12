# launcher
from optimum_benchmark.logging_utils import setup_logging
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig

# backend
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.backends.onnxruntime.config import ORTConfig

# benchmark: training & inference
from optimum_benchmark.benchmarks.inference.config import InferenceConfig

# experiment config
from optimum_benchmark.experiment import launch, ExperimentConfig


# TODO: Validate the numbers
# Models: Falcon model, LLama, Mistral, Mixtral
# Backend: Torch compile, Torch, ONNXRuntime

# Question: if the launcher should be adapted ??


if __name__ == "__main__":
    
    setup_logging(level="INFO")
    launcher_config = TorchrunConfig(nproc_per_node=2)
    benchmark_config = InferenceConfig(latency=True, memory=True, per_device_train_batch_size=1)
    backend_config = PyTorchConfig(model="gpt2", device="cuda", device_ids="0,1,2,3", no_weights=False)
    experiment_config = ExperimentConfig(
        experiment_name="api-launch",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = launch(experiment_config)
    # experiment_config.push_to_hub("TechWithRay/benchmarking") # pushes experiment_config.json to the hub
    # benchmark_report.push_to_hub("TechWithRay/benchmarking") # pushes benchmark_report.json to the hub

