from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection
from fiber.chain.models import Node

import validator.db.constants as cst
from core.constants import NETUID
from validator.core.models import RawTask
from validator.db.database import PSQLDB


async def add_task(task: RawTask, psql_db: PSQLDB) -> RawTask:
    """Add a new task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASKS_TABLE}
            ({cst.MODEL_ID}, {cst.DS_ID}, {cst.FIELD_SYSTEM}, {cst.FIELD_INSTRUCTION}, {cst.FIELD_INPUT}, {cst.STATUS},
             {cst.HOURS_TO_COMPLETE}, {cst.FIELD_OUTPUT}, {cst.FORMAT}, {cst.NO_INPUT_FORMAT}, {cst.USER_ID}, {cst.IS_ORGANIC})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING {cst.TASK_ID}
        """
        task_id = await connection.fetchval(
            query,
            task.model_id,
            task.ds_id,
            task.field_system,
            task.field_instruction,
            task.field_input,
            task.status,
            task.hours_to_complete,
            task.field_output,
            task.format,
            task.no_input_format,
            task.user_id,
            task.is_organic,
        )
        return await get_task(task_id, psql_db)


async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB) -> List[Node]:
    """Get all nodes assigned to a task for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} ON nodes.hotkey = task_nodes.hotkey
            WHERE task_nodes.task_id = $1
            AND nodes.netuid = $2
            AND task_nodes.netuid = $2
            """,
            task_id,
            NETUID,
        )
        return [Node(**dict(row)) for row in rows]


async def get_tasks_with_status(status: str, psql_db: PSQLDB, include_not_ready_tasks=False) -> List[RawTask]:
    """Get all tasks with a specific status and delay_timestamp before current time if even_not_ready is False"""

    delay_timestamp_clause = (
        "" if include_not_ready_tasks else f"AND ({cst.NEXT_DELAY_AT} IS NULL OR {cst.NEXT_DELAY_AT} <= NOW())"
    )

    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.STATUS} = $1
            {delay_timestamp_clause}
        """
        rows = await connection.fetch(query, status)
        return [RawTask(**dict(row)) for row in rows]


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    """Assign a node to a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, NETUID)


async def update_task(updated_task: RawTask, psql_db: PSQLDB) -> RawTask:
    """Update a task and its assigned miners"""
    existing_task = await get_task(updated_task.task_id, psql_db)
    if not existing_task:
        raise ValueError("Task not found")

    updates = {}
    for field, value in updated_task.dict(exclude_unset=True, exclude={"assigned_miners", "updated_timestamp"}).items():
        if getattr(existing_task, field) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            if updates:
                set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
                values = list(updates.values())
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {set_clause}{', ' if updates else ''}{cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id, *values)
            else:
                query = f"""
                    UPDATE {cst.TASKS_TABLE}
                    SET {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id)

            if updated_task.assigned_miners is not None:
                await connection.execute(
                    f"DELETE FROM {cst.TASK_NODES_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2",
                    updated_task.task_id,
                    NETUID,
                )
                if updated_task.assigned_miners:
                    # Now assuming assigned_miners is just a list of hotkeys
                    query = f"""
                        INSERT INTO {cst.TASK_NODES_TABLE} ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID})
                        SELECT $1, nodes.{cst.HOTKEY}, $3
                        FROM {cst.NODES_TABLE} nodes
                        WHERE nodes.{cst.HOTKEY} = ANY($2)
                        AND nodes.{cst.NETUID} = $3
                    """
                    await connection.execute(query, updated_task.task_id, updated_task.assigned_miners, NETUID)

    return await get_task(updated_task.task_id, psql_db)


async def get_test_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get test data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.TEST_DATA} FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)


async def get_synthetic_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get synthetic data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.SYNTHETIC_DATA} FROM {cst.TASKS_TABLE}
            WHERE {cst.TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)


async def get_tasks_ready_to_evaluate(psql_db: PSQLDB) -> List[RawTask]:
    """Get all tasks ready for evaluation"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.TASKS_TABLE}
            WHERE {cst.STATUS} = 'training'
            AND NOW() AT TIME ZONE 'UTC' > {cst.TERMINATION_AT} AT TIME ZONE 'UTC'
            AND EXISTS (
                SELECT 1 FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = {cst.TASK_ID}
                AND {cst.NETUID} = $1
            )
        """
        rows = await connection.fetch(query, NETUID)
        return [RawTask(**dict(row)) for row in rows]


async def get_tasks_by_user(user_id: str, psql_db: PSQLDB) -> List[RawTask]:
    """Get all tasks for a user"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT t.* FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
            WHERE t.{cst.USER_ID} = $1
            AND (tn.{cst.NETUID} = $2 OR tn.{cst.NETUID} IS NULL)
        """
        rows = await connection.fetch(query, user_id, NETUID)
        return [RawTask(**dict(row)) for row in rows]


async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    """Delete a task and its associated node assignments"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First delete task_nodes entries for this netuid
            await connection.execute(
                f"""
                DELETE FROM {cst.TASK_NODES_TABLE}
                WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
                """,
                task_id,
                NETUID,
            )

            # Then delete the task if it has no more node assignments
            await connection.execute(
                f"""
                DELETE FROM {cst.TASKS_TABLE}
                WHERE {cst.TASK_ID} = $1
                AND NOT EXISTS (
                    SELECT 1 FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1
                    AND {cst.NETUID} = $2
                )
                """,
                task_id,
                NETUID,
            )


async def get_miners_for_task(task_id: UUID, psql_db: PSQLDB) -> List[Node]:
    """Retrieve all miners assigned to a specific task."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT nodes.* FROM {cst.NODES_TABLE} nodes
            JOIN {cst.TASK_NODES_TABLE} task_nodes
            ON nodes.hotkey = task_nodes.hotkey AND nodes.netuid = task_nodes.netuid
            WHERE task_nodes.task_id = $1
        """
        rows = await connection.fetch(query, task_id)
        return [Node(**dict(row)) for row in rows]


async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[RawTask]:
    """Get a task by ID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if row:
            return RawTask(**dict(row))
        return None


async def get_winning_submissions_for_task(task_id: UUID, psql_db: PSQLDB) -> List[Dict]:
    """Retrieve the winning submission for a task based on the highest quality score in task_nodes."""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT submissions.repo
            FROM {cst.SUBMISSIONS_TABLE} submissions
            JOIN {cst.TASK_NODES_TABLE} task_nodes
            ON submissions.task_id = task_nodes.task_id
            AND submissions.hotkey = task_nodes.hotkey
            AND submissions.netuid = task_nodes.netuid
            WHERE submissions.task_id = $1
            ORDER BY task_nodes.quality_score DESC
            LIMIT 1
        """
        rows = await connection.fetch(query, task_id)
        return [dict(row) for row in rows]


async def get_task_by_id(task_id: UUID, psql_db: PSQLDB) -> RawTask:
    """Get a task by ID along with its winning submissions"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT submissions.task_id, submissions.repo
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                ON submissions.task_id = task_nodes.task_id
                AND submissions.hotkey = task_nodes.hotkey
                AND submissions.netuid = task_nodes.netuid
                WHERE submissions.task_id = $1
                AND task_nodes.quality_score IS NOT NULL
                ORDER BY task_nodes.quality_score DESC
                LIMIT 1
            )
            SELECT
                tasks.*,
                victorious_repo.repo as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
            WHERE tasks.{cst.TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if not row:
            return None

        return RawTask(**dict(row))


async def get_tasks(psql_db: PSQLDB, limit: int = 100, offset: int = 0) -> List[RawTask]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            WITH victorious_repo AS (
                SELECT submissions.task_id, submissions.repo
                FROM {cst.SUBMISSIONS_TABLE} submissions
                JOIN {cst.TASK_NODES_TABLE} task_nodes
                ON submissions.task_id = task_nodes.task_id
                AND submissions.hotkey = task_nodes.hotkey
                AND submissions.netuid = task_nodes.netuid
                WHERE submissions.task_id = $1
                AND task_nodes.quality_score IS NOT NULL
                ORDER BY task_nodes.quality_score DESC
                LIMIT 1
            )
            SELECT
                tasks.*,
                victorious_repo.repo as trained_model_repository
            FROM {cst.TASKS_TABLE} tasks
            LEFT JOIN victorious_repo ON tasks.task_id = victorious_repo.task_id
            ORDER BY tasks.created_timestamp DESC
            LIMIT $1 OFFSET $2
        """


        rows = await connection.fetch(query, limit, offset)
        return [RawTask(**dict(row)) for row in rows]
