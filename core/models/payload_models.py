from uuid import UUID

from fiber.logging_utils import get_logger
from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import JobStatus
from core.models.utility_models import TaskStatus


logger = get_logger(__name__)


class MinerTaskRequest(BaseModel):
    ds_size: int
    model: str
    hours_to_complete: int
    task_id: str


class TrainRequest(BaseModel):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    task_id: str
    hours_to_complete: int


class TrainResponse(BaseModel):
    message: str
    task_id: UUID


class JobStatusPayload(BaseModel):
    task_id: UUID


class JobStatusResponse(BaseModel):
    task_id: UUID
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str


class EvaluationResult(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class DatasetColumnsResponse(BaseModel):
    instruction_col: str
    input_col: str | None = None
    output_col: str | None = None
    system_col: str | None = None


class CreateTaskRequest(BaseModel):
    instruction_col: str = Field(..., description="The column name for the instruction", examples=["instruction"])
    input_col: str | None = Field(None, description="The column name for the input", examples=["input"])
    output_col: str | None = Field(None, description="The column name for the output", examples=["output"])
    system_col: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["HuggingFaceFW/fineweb-2"])
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])
    hours_to_complete: int = Field(..., description="The number of hours to complete the task", examples=[1])

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class CreateTaskResponse(BaseModel):
    success: bool = Field(..., description="Whether the task was created successfully")
    task_id: str = Field(..., description="The ID of the task")
    message: str | None = Field(None, description="The message from the task creation")


class SubmitTaskSubmissionRequest(BaseModel):
    task_id: str
    node_id: int
    repo: str


class SubmissionResponse(BaseModel):
    success: bool
    message: str
    submission_id: str | None = None


class NewTaskRequest(BaseModel):
    model_repo: str
    ds_repo: str
    instruction_col: str
    input_col: str | None = None
    hours_to_complete: int
    system_col: str | None = None
    output_col: str | None = None
    format_col: str | None = None
    no_input_format_col: str | None = None

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class GetTasksRequest(BaseModel):
    fingerprint: str


class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID | None = None


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float


class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


class AllOfNodeResults(BaseModel):
    success: bool
    hotkey: str
    task_results: list[TaskMinerResult] | None


class TaskResultResponse(BaseModel):
    success: bool
    id: UUID
    miner_results: list[MinerTaskResult] | None


class TaskStatusResponse(BaseModel):
    success: bool
    id: UUID
    status: TaskStatus
    miners: list[dict] | None  # TODO: Improve with actual types
    model_repo: str
    ds_repo: str | None
    input_col: str | None
    system_col: str | None
    output_col: str | None
    instruction_col: str
    format_col: str | None
    no_input_format_col: str | None
    started: str
    end: str
    created: str
    hours_to_complete: int
    winning_submission: WinningSubmission | None = None

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus
