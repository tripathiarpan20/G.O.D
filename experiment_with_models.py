import os
from dataclasses import dataclass
from typing import Any
from datetime import datetime
from typing import Optional

import docker
import requests
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_model_info
from core.docker_utils import stream_logs
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import Job

from validator.utils.call_endpoint import process_non_stream_get
import validator.core.constants as csts

import csv
import json
import subprocess
import traceback


logger = get_logger(__name__)


@dataclass
class DockerEnvironment:
    job_id: str
    dataset_type: str
    dataset_filename: str
    huggingface_token: str | None = None
    wandb_token: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    task_id: str,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """

    try:
        architecture, model_type = get_architecture_and_model_type(model)
    except Exception as e:
        logger.error(f"Error getting architecture and model type for {model}: {e}")
        architecture = None
        model_type = None

    list_of_models_with_flash_attention = ["llama", "gemma", "mistral", "jamba"]
    list_of_model_names_wo_flash_attention = ["NousResearch/CodeLlama", "NousResearch/Yarn-Llama"]





    logger.info("Loading config template")
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    config = update_model_info(config, model, task_id)
    config["mlflow_experiment_name"] = dataset

    if architecture is not None and model_type is not None:
        # try to match substrings in lowercase
        if any(substring in architecture.lower() for substring in list_of_models_with_flash_attention) or any(substring in model_type.lower() for substring in list_of_models_with_flash_attention):
            logger.info(f"Model {model} has flash attention")
            config["flash_attention"] = True
    else:
        if any(substring.lower() in model.lower() for substring in list_of_model_names_wo_flash_attention):
            logger.info(f"Model {model} does not have flash attention")
            config["flash_attention"] = False
        #try to match substrings in lowercase for model
        elif any(substring in model.lower() for substring in list_of_models_with_flash_attention):
            logger.info(f"Model {model} has flash attention")
            config["flash_attention"] = True

    
    
    return config

def get_num_params(model_id: str) -> int:
    """
    Get the number of parameters of a model
    """
    # use curl 'https://dataset-model-checker-1.onrender.com/models?search=unsloth%2Fmistral-7b-instruct-v0.3' \
    #   --header 'Authorization: test_key'
    api_url = f"https://dataset-model-checker-1.onrender.com/models?search={model_id}"
    response = requests.get(api_url)
    return response.json()[0]["parameter_count"]

def get_architecture_and_model_type(model_id: str) -> str:
    """
    Get the architecture of a model
    """
    api_url = f"https://dataset-model-checker-1.onrender.com/models?search={model_id}"
    response = requests.get(api_url)
    architecture = response.json()[0]["config"]["architectures"][0]
    model_type = response.json()[0]["config"]["model_type"]
    return architecture, model_type

def all_models_list() -> list[dict[str, Any]]:
    """
    Returns a list of all available model IDs and their metadata from the models endpoint using curl.
    """
    result = subprocess.run(
        [
            "curl",
            "https://dataset-model-checker-1.onrender.com/models/random",
            "--header",
            "Authorization: test_key"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Parse the JSON response
    response = json.loads(result.stdout)
    if not isinstance(response, list):
        raise TypeError("Expected a list of responses from the endpoint")
    
    # Extract model_id and metadata (formerly config)
    model_data = [{"model_id": model["model_id"], "metadata": model["config"]} for model in response]
    
    return model_data


@dataclass
class JobResult:
    model: str
    status: str
    error_message: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    config: Optional[dict]
    metadata: Optional[dict]

def _write_result_to_json(result: JobResult, total_models: int, current_index: int, file_path: str):
    """
    Append a single job result to the JSON file, creating the file if it doesn't exist.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {
            'total_models': total_models,
            'current_model_index': 0,
            'results': []
        }
    
    result_dict = {
        'run_number': len(data['results']) + 1,
        'model': result.model,
        'status': result.status,
        'error_message': result.error_message or '',
        'start_time': result.start_time.isoformat(),
        'end_time': result.end_time.isoformat() if result.end_time else '',
        'duration_seconds': f"{result.duration:.2f}" if result.duration else '',
        'config': result.config,
        'metadata': result.metadata
    }
    
    data['results'].append(result_dict)
    data['current_model_index'] = current_index
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def _write_error_to_csv(model_id: str, error_message: str, duration: float, csv_path: str):
    """
    Write error information to a CSV file.
    """
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model_id', 'error_message', 'duration_seconds'])
        
        writer.writerow([
            datetime.now().isoformat(),
            model_id,
            error_message,
            f"{duration:.2f}"
        ])

def create_debug_jobs() -> str:
    """
    Development/debugging function to create and run jobs for all available models
    and generate a JSON report of the results.
    """
    models = all_models_list()
    total_models = len(models)
    
    main_path = os.path.join(cst.OUTPUT_DIR, "debug_jobs_report.json")
    backup_path = os.path.join(cst.OUTPUT_DIR, "debug_jobs_report_backup.json")
    error_csv_path = os.path.join(cst.OUTPUT_DIR, "debug_jobs_errors.csv")
    
    logger.info("=" * 80)
    logger.info(f"Starting debug run for {total_models} models")
    logger.info("=" * 80)
    
    dataset_type = CustomDatasetType(
        field_input="input",
        field_instruction="instruction",
        field_output="output"
    )
    
    for index, model in enumerate(models, 1):
        logger.info("=" * 80)
        logger.info(f"Processing model {index}/{total_models}: {model}")
        logger.info(f"Progress: {(index/total_models)*100:.1f}%")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        job_result = JobResult(
            model=model['model_id'],
            status="STARTED",
            error_message=None,
            start_time=start_time,
            end_time=None,
            duration=None,
            config=None,
            metadata=model.get('metadata')
        )
        
        try:
            job = Job(
                dataset="tmp_ds/alpaca_samples.json",
                model=model['model_id'],
                dataset_type=dataset_type,
                file_format=FileFormat.JSON
            )
            
            config = _load_and_modify_config(
                job.dataset, 
                job.model, 
                job.dataset_type, 
                job.file_format, 
                job.job_id
            )
            job_result.config = config
            
            start_tuning_container_dev(job)
            
            job_result.status = "SUCCESS"
            logger.info(f"✅ Model {model} completed successfully")
            
        except DockerException as e:
            job_result.status = "FAILED"
            job_result.error_message = str(e)
            logger.error(f"❌ Container error processing model {model}")
        except Exception as e:
            job_result.status = "FAILED"
            job_result.error_message = "Host Machine Error - " + ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.exception(f"❌ Host machine error processing model {model}")
        
        finally:
            job_result.end_time = datetime.now()
            job_result.duration = (job_result.end_time - job_result.start_time).total_seconds()
            
            # Write result to JSON
            _write_result_to_json(job_result, total_models, index, main_path)
            
            # If there was an error, write it to CSV
            if job_result.status == "FAILED":
                _write_error_to_csv(
                    model_id=job_result.model,
                    error_message=job_result.error_message or "Unknown error",
                    duration=job_result.duration,
                    csv_path=error_csv_path
                )
            
            # Create backup every 10 entries
            if index % 10 == 0:
                import shutil
                shutil.copy2(main_path, backup_path)
                logger.info(f"Backup created at {backup_path}")
            
            logger.info(f"Time taken for this model: {job_result.duration:.1f} seconds")
    
    logger.info("=" * 80)
    logger.info(f"Debug run completed for all {total_models} models")
    logger.info(f"Final results available at: {main_path}")
    logger.info(f"Error log available at: {error_csv_path}")
    logger.info("=" * 80)
    
    return main_path


def start_tuning_container_dev(job: Job):
    """
    Development version of start_tuning_container with the same functionality,
    but skipping model publishing to HuggingFace and wandb logging.
    """
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER (DEV VERSION)")
    logger.info("=" * 80)

    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(job.dataset, job.model, job.dataset_type, job.file_format, job.job_id)
    
    # Remove HuggingFace-related settings
    config.pop('hub_model_id', None)
    config.pop('hub_repo', None)
    config.pop('hub_strategy', None)
    config.pop('hub_token', None)
    
    # Also remove wandb settings since we're in dev mode
    config.pop('wandb_project', None)
    config.pop('wandb_entity', None)
    config.pop('wandb_mode', None)
    config.pop('wandb_run', None)
    config.pop('wandb_runid', None)
    config.pop('wandb_name', None)
    
    save_config(config, config_path)
    logger.info(config)

    # Always treat as local file for dev version
    dataset_filename = os.path.basename(job.dataset)
    logger.info(f"Using local dataset: {dataset_filename}")

    docker_env = DockerEnvironment(
        job_id=job.job_id,
        dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
        dataset_filename=dataset_filename,  # Always use the local filename
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
            os.path.dirname(os.path.abspath(job.dataset)): {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }
        }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Stream logs in real-time
        stream_logs(container)

        # Get the container's exit code and capture all logs
        result = container.wait()
        logs = container.logs(stdout=True, stderr=True).decode('utf-8')
        
        if result["StatusCode"] != 0:
            error_msg = f"Container failed with exit code {result['StatusCode']}.\nContainer logs:\n{logs}"
            raise DockerException(error_msg)
        else:
            logger.info("Container completed successfully. Full logs:")
            logger.info(logs)

    except DockerException as e:
        # Pass through Docker container errors without the host machine prefix
        logger.error(f"Container error: {str(e)}")
        raise
    except Exception as e:
        # Only add host machine prefix for actual host machine errors
        error_traceback = "Host Machine Error - " + ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"Error processing job: {error_traceback}")
        raise Exception(error_traceback)

    finally:
        if "container" in locals():
            container.remove(force=True)

def main():
    """
    Main function to run the debug experiments.
    """
    try:
        logger.info("Starting debug experiments...")
        json_path = create_debug_jobs()
        logger.info(f"Experiments completed successfully. Results available at: {json_path}")
        
    except Exception as e:
        logger.error("Failed to complete experiments")
        logger.exception(e)
        raise

def run_single_experiment(model_id: str = "NousResearch/GPT4-x-Vicuna-13b-fp16") -> None:
    """
    Run a single experiment with a specified model.
    
    Args:
        model_id: The HuggingFace model ID to test
    """
    logger.info("=" * 80)
    logger.info(f"Starting single experiment with model: {model_id}")
    logger.info("=" * 80)
    
    # Use the same dataset configuration as in create_debug_jobs
    dataset_type = CustomDatasetType(
        field_input="input",
        field_instruction="instruction",
        field_output="output"
    )
    
    try:
        job = Job(
            dataset="tmp_ds/alpaca_samples.json",
            model=model_id,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON
        )
        
        # Get the config and log it for inspection
        config = _load_and_modify_config(
            job.dataset, 
            job.model, 
            job.dataset_type, 
            job.file_format, 
            job.job_id
        )
        logger.info("Configuration to be used:")
        logger.info(config)
        
        # Start the container for this job
        start_tuning_container_dev(job)
        
        logger.info(f"✅ Experiment with model {model_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error running experiment with model {model_id}")
        logger.exception(e)
        raise

def run_model_experiments(json_path: str = "tmp_res/results_processed.json") -> None:
    """
    Run experiments for a predefined list of models and update their error messages in the JSON file.
    
    Args:
        json_path: Path to the JSON file containing results
    """
    models_to_test = [
        "migtissera/Tess-v2.5-Phi-3-medium-128k-14B",
        "NousResearch/Yarn-Llama-2-13b-64k",
        "teknium/OpenHermes-2.5-Mistral-7B",
        "unsloth/mistral-7b-v0.3",
        "unsloth/mistral-7b-instruct-v0.3",
        "MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4",
        "NousResearch/Yarn-Llama-2-7b-128k",
        "unsloth/OpenHermes-2.5-Mistral-7B",
        "heegyu/WizardVicuna2-13b-hf",
        "NousResearch/Nous-Hermes-2-SOLAR-10.7B",
        "lmsys/vicuna-7b-v1.5",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "JackFram/llama-68m",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "unsloth/Qwen2.5-1.5B",
        "NousResearch/Yarn-Llama-2-13b-128k",
        "unsloth/mistral-7b-v0.2",
        "NousResearch/GPT4-x-Vicuna-13b-fp16",
        "dltjdgh0928/test_instruction",
        "NousResearch/Yarn-Llama-2-7b-64k",
        "lmsys/vicuna-7b-v1.3",
        "openlm-research/open_llama_3b",
        "lmsys/vicuna-13b-v1.5",
        "01-ai/Yi-1.5-9B-Chat-16K"
    ]
    
    # Read existing JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    for model_id in models_to_test:
        logger.info("=" * 80)
        logger.info(f"Processing model: {model_id}")
        logger.info("=" * 80)
        
        try:
            # Find the corresponding result entry
            result_entry = next((r for r in results if r['model'] == model_id), None)
            if not result_entry:
                logger.warning(f"No existing entry found for model {model_id}")
                continue
            
            job = Job(
                dataset="tmp_ds/alpaca_samples.json",
                model=model_id,
                dataset_type=CustomDatasetType(
                    field_input="input",
                    field_instruction="instruction",
                    field_output="output"
                ),
                file_format=FileFormat.JSON
            )
            
    
            config_filename = f"{job.job_id}.yml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)

            config = _load_and_modify_config(
                job.dataset, 
                job.model, 
                job.dataset_type, 
                job.file_format, 
                job.job_id
            )
            # Remove HuggingFace-related settings
            config.pop('hub_model_id', None)
            config.pop('hub_repo', None)
            config.pop('hub_strategy', None)
            config.pop('hub_token', None)
            
            # Also remove wandb settings since we're in dev mode
            config.pop('wandb_project', None)
            config.pop('wandb_entity', None)
            config.pop('wandb_mode', None)
            config.pop('wandb_run', None)
            config.pop('wandb_runid', None)
            config.pop('wandb_name', None)
            
            save_config(config, config_path)
            logger.info(config)

            # Always treat as local file for dev version
            dataset_filename = os.path.basename(job.dataset)
            logger.info(f"Using local dataset: {dataset_filename}")


            docker_env = DockerEnvironment(
                job_id=job.job_id,
                dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
                dataset_filename=os.path.basename(job.dataset),
            ).to_dict()
            
            docker_client = docker.from_env()
            volume_bindings = {
                os.path.abspath(cst.CONFIG_DIR): {
                    "bind": "/workspace/axolotl/configs",
                    "mode": "rw",
                },
                os.path.abspath(cst.OUTPUT_DIR): {
                    "bind": "/workspace/axolotl/outputs",
                    "mode": "rw",
                },
                os.path.dirname(os.path.abspath(job.dataset)): {
                    "bind": "/workspace/input_data",
                    "mode": "ro",
                }
            }
            
            container = docker_client.containers.run(
                image=cst.MINER_DOCKER_IMAGE,
                environment=docker_env,
                volumes=volume_bindings,
                runtime="nvidia",
                device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
                detach=True,
                tty=True,
            )

            # Stream logs in real-time
            stream_logs(container)

            # Wait for completion and get logs
            result = container.wait()
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            if result["StatusCode"] != 0:
                error_msg = f"Container failed with exit code {result['StatusCode']}.\nContainer logs:\n{logs}"
                result_entry['error_message'] = error_msg
            else:
                result_entry['error_message'] = ''
            
        except Exception as e:
            if result_entry:
                # Capture the full traceback as a string
                error_traceback = "Host Machine Error - " + ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                result_entry['error_message'] = error_traceback
            logger.exception(f"Error processing model {model_id}")
        
        finally:
            # Save updated results after each model
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            if 'container' in locals():
                container.remove(force=True)
            
            logger.info(f"Completed processing {model_id}")

def retry_failed_models(json_path: str = "tmp_res/results_processed.json") -> None:
    """
    Retry experiments for previously failed models, using the same container setup as run_model_experiments.
    Results are tracked in JSON and individual log files for each model.
    
    Args:
        json_path: Path to the JSON file containing previous results
    """
    # Read existing JSON file
    with open(json_path, 'r') as f:
        original_data = json.load(f)
    
    # Filter for failed models
    failed_results = [r for r in original_data['results'] if r['status'] == "FAILED"]
    
    # Setup paths for retry results and logs
    retry_json_path = json_path.replace('.json', '_retry.json')
    logs_base_dir = os.path.join(cst.OUTPUT_DIR, "retry_logs")
    success_logs_dir = os.path.join(logs_base_dir, "success")
    failed_logs_dir = os.path.join(logs_base_dir, "failed")
    
    # Create log directories
    os.makedirs(success_logs_dir, exist_ok=True)
    os.makedirs(failed_logs_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"Starting retry run for {len(failed_results)} failed models")
    logger.info(f"Results will be saved to: {retry_json_path}")
    logger.info(f"Success logs will be saved in: {success_logs_dir}")
    logger.info(f"Failure logs will be saved in: {failed_logs_dir}")
    logger.info("=" * 80)
    
    dataset_type = CustomDatasetType(
        field_input="input",
        field_instruction="instruction",
        field_output="output"
    )
    
    for index, result in enumerate(failed_results, 1):
        model_id = result['model']
        safe_model_name = model_id.replace('/', '_').replace('\\', '_')
        temp_log_file = os.path.join(logs_base_dir, f"{safe_model_name}.log")
        
        existing_log_path_success = os.path.join(success_logs_dir, f"{safe_model_name}.log")
        existing_log_path_failed = os.path.join(failed_logs_dir, f"{safe_model_name}.log")
        
        if os.path.exists(existing_log_path_success) or os.path.exists(existing_log_path_failed):
            logger.info(f"Skipping model {model_id} as logs already exist.")
            continue

        logger.info("=" * 80)
        logger.info(f"Retrying model {index}/{len(failed_results)}: {model_id}")
        logger.info(f"Previous error: {result.get('error_message', 'No error recorded')[:200]}...")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            job = Job(
                dataset="tmp_ds/alpaca_samples.json",
                model=model_id,
                dataset_type=dataset_type,
                file_format=FileFormat.JSON
            )
            
            config = _load_and_modify_config(
                job.dataset, 
                job.model, 
                job.dataset_type, 
                job.file_format, 
                job.job_id
            )
            
            # Remove HuggingFace and wandb settings
            for key in ['hub_model_id', 'hub_repo', 'hub_strategy', 'hub_token',
                       'wandb_project', 'wandb_entity', 'wandb_mode', 'wandb_run',
                       'wandb_runid', 'wandb_name']:
                config.pop(key, None)
            
            save_config(config, os.path.join(cst.CONFIG_DIR, f"{job.job_id}.yml"))
            
            docker_env = DockerEnvironment(
                job_id=job.job_id,
                dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
                dataset_filename=os.path.basename(job.dataset),
            ).to_dict()
            
            docker_client = docker.from_env()
            volume_bindings = {
                os.path.abspath(cst.CONFIG_DIR): {
                    "bind": "/workspace/axolotl/configs",
                    "mode": "rw",
                },
                os.path.abspath(cst.OUTPUT_DIR): {
                    "bind": "/workspace/axolotl/outputs",
                    "mode": "rw",
                },
                os.path.dirname(os.path.abspath(job.dataset)): {
                    "bind": "/workspace/input_data",
                    "mode": "ro",
                }
            }
            
            container = docker_client.containers.run(
                image=cst.MINER_DOCKER_IMAGE,
                environment=docker_env,
                volumes=volume_bindings,
                runtime="nvidia",
                device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
                detach=True,
                tty=True,
            )

            # Write logs to temporary file first
            with open(temp_log_file, 'w') as f:
                f.write(f"Retry attempt for model: {model_id}\n")
                f.write(f"Start time: {start_time.isoformat()}\n")
                f.write(f"Previous error: {result.get('error_message', 'No error recorded')}\n")
                f.write("\nContainer logs:\n")
                f.write("=" * 80 + "\n\n")
                
                for log in container.logs(stream=True, stdout=True, stderr=True):
                    try:
                        log_line = log.decode('utf-8')
                    except UnicodeDecodeError:
                        log_line = log.decode('utf-8', errors='replace')
                    except:
                        log_line = "[LOG DECODE ERROR]\n"
                    
                    f.write(log_line)
                    print(log_line, end='')

            # Get container result
            container_result = container.wait()
            
            if container_result["StatusCode"] != 0:
                final_log_path = os.path.join(failed_logs_dir, f"{safe_model_name}.log")
                error_msg = f"Container failed with exit code {container_result['StatusCode']}"
                result['error_message'] = error_msg
                result['status'] = "FAILED"
                
                # Append error message to log
                with open(temp_log_file, 'a') as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Error: {error_msg}\n")
            else:
                final_log_path = os.path.join(success_logs_dir, f"{safe_model_name}.log")
                result['error_message'] = ''
                result['status'] = "SUCCESS"
                logger.info(f"✅ Model {model_id} completed successfully")
            
        except Exception as e:
            final_log_path = os.path.join(failed_logs_dir, f"{safe_model_name}.log")
            error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            result['error_message'] = error_traceback
            result['status'] = "FAILED"
            
            # Append error to log
            with open(temp_log_file, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Exception occurred:\n")
                f.write(error_traceback)
            
            logger.exception(f"Error processing model {model_id}")
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            with open(temp_log_file, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"End time: {end_time.isoformat()}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Final status: {result['status']}\n")
            
            if os.path.exists(temp_log_file):
                import shutil
                shutil.move(temp_log_file, final_log_path)
            
            result['start_time'] = start_time.isoformat()
            result['end_time'] = end_time.isoformat()
            result['duration_seconds'] = f"{duration:.2f}"
            
            with open(retry_json_path, 'w') as f:
                json.dump(original_data, f, indent=2)
            
            if 'container' in locals():
                container.remove(force=True)
            
            logger.info(f"Time taken for this model: {duration:.1f} seconds")
            logger.info(f"Log saved to: {final_log_path}")
    
    successful_retries = len([r for r in failed_results if r['status'] == "SUCCESS"])
    logger.info("=" * 80)
    logger.info("Retry Run Summary:")
    logger.info(f"Total models retried: {len(failed_results)}")
    logger.info(f"Successful retries: {successful_retries}")
    logger.info(f"Failed retries: {len(failed_results) - successful_retries}")
    logger.info(f"Results saved to: {retry_json_path}")
    logger.info(f"Success logs saved in: {success_logs_dir}")
    logger.info(f"Failure logs saved in: {failed_logs_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Run the main function directly
    # run_single_experiment("NousResearch/GPT4-x-Vicuna-13b-fp16")
    # run_model_experiments()
    # main()
    retry_failed_models()

