import json
from datetime import datetime
from pathlib import Path
from typing import Any

from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


class ModelConfig(BaseModel):
    architectures: list[str]
    model_type: str


class DatasetData(BaseModel):
    dataset_id: str
    sparse_columns: list[str] = Field(default_factory=list)
    non_sparse_columns: list[str] = Field(default_factory=list)


class ModelData(BaseModel):
    model_id: str
    parameter_count: int | None = None


class Task(BaseModel):
    is_organic: bool
    task_id: UUID | None = None
    model_id: str
    ds_id: str
    input: str | None = None
    status: str
    system: str | None = None
    instruction: str | None = None
    output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    test_data: str | None = None
    synthetic_data: str | None = None
    training_data: str | None = None
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    created_timestamp: datetime | None = None
    delay_timestamp: datetime | None = None
    delay_times: int | None = 0
    updated_timestamp: datetime | None = None
    started_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    completed_timestamp: datetime | None = None
    hours_to_complete: int
    best_submission_repo: str | None = None
    user_id: str | None = None

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    hotkey: str
    normalised_score: float | None = 0.0


class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float


class TaskResults(BaseModel):
    task: Task
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    task_work_scores: list[float] = Field(default_factory=list)
    average_raw_score: float | None = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: float | None = Field(default=0.0)
    emission: float | None = Field(default=0.0)
    task_raw_scores: list[float] = Field(default_factory=list)
    hotkey: str

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: float | None = None
    task_id: UUID
    hotkey: str
    repo: str
    created_on: datetime | None = None
    updated_on: datetime | None = None


class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: float | None = 0.0
    submission: Submission | None = None


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


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats


class DatasetUrls(BaseModel):
    test_url: str
    synthetic_url: str | None = None
    train_url: str


class DatasetFiles(BaseModel):
    prefix: str
    data: str
    temp_path: Path | None = None


class DatasetJsons(BaseModel):
    train_data: list[Any]
    test_data: list[Any]
    synthetic_data: list[Any] = Field(default_factory=list)

    def to_json_strings(self) -> dict[str, str]:
        return {
            'train_data': json.dumps(self.train_data),
            'test_data': json.dumps(self.test_data),
            'synthetic_data': json.dumps(self.synthetic_data) if self.synthetic_data else ""
        }
