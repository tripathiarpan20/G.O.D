import asyncio
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator

from fiber.logging_utils import get_logger
from substrateinterface import Keypair

import validator.core.constants as csts
from core.models.payload_models import DatasetRequest
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import DatasetData
from validator.core.models import Task
from validator.core.models import ModelData
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_tasks_with_status
from validator.utils.call_endpoint import call_content_service


logger = get_logger(name="task synth")


async def _get_models(keypair: Keypair) -> AsyncGenerator[str, None]:
    response: list[dict[str, Any]] = await call_content_service(csts.GET_RANDOM_MODELS_ENDPOINT, keypair)
    logger.info(f"Get model response was {response}")
    models = [ModelData(**model) for model in response]
    random.shuffle(models)
    for m in models:
        yield m.model_id


async def _get_datasets(keypair: Keypair) -> AsyncGenerator[str, None]:
    response: list[dict[str, Any]] = await call_content_service(csts.GET_RANDOM_DATASETS_ENDPOINT, keypair)
    logger.info(f"Get dataset response was {response}")
    datasets = [DatasetData(**ds) for ds in response]
    random.shuffle(datasets)
    for ds in datasets:
        yield ds.dataset_id


async def _get_the_columns_for_dataset(dataset_id: str, keypair: Keypair) -> DatasetRequest:
    url = csts.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace(
        "{dataset}", dataset_id)
    response = await call_content_service(url, keypair)
    if not isinstance(response, dict):
        raise TypeError(f"Expected dictionary response, got {type(response)}")
    try:
        columns = DatasetRequest.model_validate(response)
    except Exception as exc:
        raise TypeError(
            f"The get columns for dataset endpoint should return a DatasetRequest type: {exc}")
    return columns


async def create_a_new_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[str, None],
):
    number_of_hours = random.randint(
        csts.MIN_COMPETITION_HOURS, csts.MAX_COMPETITION_HOURS)
    model_id = await anext(models)
    dataset_id = await anext(datasets)
    columns = await _get_the_columns_for_dataset(dataset_id, config.keypair)
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = Task(
        model_id=model_id,
        ds_id=dataset_id,
        system=None,
        instruction=columns.instruction_col,
        input=columns.input_col,
        output=columns.output_col,
        status=TaskStatus.PENDING,
        is_organic=False,
        end_timestamp=end_timestamp,
        hours_to_complete=number_of_hours,
    )
    logger.info(f"New task created and added to the queue {task}")

    task = await add_task(task, config.psql_db)


async def _add_new_task_to_network_if_not_enough(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[str, None],
):
    current_training_tasks = await get_tasks_with_status(TaskStatus.TRAINING, config.psql_db)
    current_delayed_tasks = await get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    logger.info(f"We have {(len(current_delayed_tasks))} tasks in the queue")
    logger.info(
        f"There are {len(current_training_tasks)} running at the moment")
    if len(current_delayed_tasks) == 0 and len(current_training_tasks) < csts.HOW_MANY_TASKS_MINIMAL_AT_THE_SAME_TIME:
        logger.info("This is less than the minimal - creating a new task")
        await create_a_new_task(config, models, datasets)


async def synthetic_task_loop(config: Config):
    logger.info("Starting the synthetic task loop")
    datasets = _get_datasets(config.keypair)
    models = _get_models(config.keypair)
    while True:
        try:
            await _add_new_task_to_network_if_not_enough(config, models, datasets)
            await asyncio.sleep(csts.NUMBER_OF_MINUTES_BETWEEN_SYNTH_TASK_CHECK * 60)
        except Exception as e:
            logger.info(
                f"Ah, that dataset was missing some details, trying another one next time. {e}")

            await asyncio.sleep(5 * 60)
