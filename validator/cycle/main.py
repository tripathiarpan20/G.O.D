import asyncio

from validator.core.config import Config
from validator.core.refresh_nodes import refresh_nodes_periodically
from validator.core.weight_setting import set_weights_periodically
from validator.cycle.process_tasks import process_completed_tasks
from validator.cycle.process_tasks import process_pending_tasks
from validator.tasks.synthetic_scheduler import schedule_synthetics_periodically
from validator.utils.call_endpoint import sign_up_cron_job


async def run_validator_cycles(config: Config) -> None:
    await asyncio.gather(
        refresh_nodes_periodically(config),
        schedule_synthetics_periodically(config),
        set_weights_periodically(config),
        process_completed_tasks(config),
        process_pending_tasks(config),
        sign_up_cron_job(config.keypair),
    )
