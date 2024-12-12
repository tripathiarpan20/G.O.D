import json
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class RawTask(BaseModel):
    is_organic: bool
    task_id: UUID | None = None
    model_id: str
    ds_id: str
    status: str
    account_id: UUID
    times_delayed: int = 0
    hours_to_complete: int
    field_system: str | None = None
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: None = None  # NOTE: Needs updating to be optional once we accept it
    test_data: str | None = None
    synthetic_data: str | None = None
    training_data: str | None = None
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None

    created_at: datetime
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None



    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


# NOTE: As time goes on we will expand this class to be more of a 'submmited task'?
# Might wanna rename this some more
class Task(RawTask):
    trained_model_repository: str | None = None


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    hotkey: str
    normalised_score: Optional[float] = 0.0


class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float


class TaskResults(BaseModel):
    task: RawTask
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    task_work_scores: list[float] = Field(default_factory=list)
    average_raw_score: Optional[float] = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: Optional[float] = Field(default=0.0)
    emission: Optional[float] = Field(default=0.0)
    task_raw_scores: list[float] = Field(default_factory=list)
    hotkey: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: Optional[float] = None
    task_id: UUID
    hotkey: str
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: Optional[float] = 0.0
    submission: Optional[Submission] = None


class QualityMetrics(BaseModel):
    total_score: float
    total_count: int
    total_success: int
    total_quality: int
    avg_quality_score: float = Field(ge=0.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    quality_rate: float = Field(ge=0.0, le=1.0)


class WorkloadMetrics(BaseModel):
    competition_hours: int = Field(ge=0)
    total_params_billions: float = Field(ge=0.0)


class ModelMetrics(BaseModel):
    modal_model: str
    unique_models: int = Field(ge=0)
    unique_datasets: int = Field(ge=0)


class NodeStats(BaseModel):
    quality_metrics: QualityMetrics
    workload_metrics: WorkloadMetrics
    model_metrics: ModelMetrics


class AllNodeStats(BaseModel):
    daily: NodeStats
    three_day: NodeStats
    weekly: NodeStats
    monthly: NodeStats
    all_time: NodeStats


class DatasetUrls(BaseModel):
    test_url: str
    synthetic_url: Optional[str] = None
    train_url: str


class DatasetFiles(BaseModel):
    prefix: str
    data: str
    temp_path: Optional[Path] = None


class DatasetJsons(BaseModel):
    train_data: list[Any]
    test_data: list[Any]
    synthetic_data: list[Any] = Field(default_factory=list)

    def to_json_strings(self) -> dict[str, str]:
        return {
            "train_data": json.dumps(self.train_data),
            "test_data": json.dumps(self.test_data),
            "synthetic_data": json.dumps(self.synthetic_data) if self.synthetic_data else "",
        }
