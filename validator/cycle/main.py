import asyncio

from fiber.logging_utils import get_logger

from validator.core.config import load_config
from validator.core.refresh_nodes import refresh_nodes_periodically
from validator.core.weight_setting import set_weights_periodically
from validator.cycle.process_tasks import process_completed_tasks
from validator.cycle.process_tasks import process_pending_tasks
from validator.tasks.synthetic_scheduler import schedule_synthetics_periodically
from validator.utils.call_endpoint import sign_up_cron_job
from validator.utils.util import try_db_connections


logger = get_logger(__name__)


async def run_validator_cycles() -> None:
    config = load_config()
    await try_db_connections(config)

    await asyncio.gather(
        refresh_nodes_periodically(config),
        schedule_synthetics_periodically(config),
        set_weights_periodically(config),
        process_completed_tasks(config),
        process_pending_tasks(config),
        sign_up_cron_job(config.keypair),
    )


if __name__ == "__main__":
    asyncio.run(run_validator_cycles())
