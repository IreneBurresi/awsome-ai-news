"""Repository and Git-related data models for Step 7."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class CommitInfo(BaseModel):
    """Git commit information."""

    commit_hash: str = Field(description="Git commit hash")
    message: str = Field(description="Commit message")
    timestamp: datetime = Field(description="Commit timestamp")
    author: str = Field(default="awesome-ai-news-bot", description="Commit author")
    files_changed: int = Field(ge=0, description="Number of files changed")


class Step7Result(BaseModel):
    """Result from Step 7: Repository update."""

    success: bool = Field(description="Whether step completed successfully")
    readme_updated: bool = Field(default=False, description="Whether README was updated")
    news_file_created: Path | None = Field(
        default=None, description="Path to created news YAML file"
    )
    archive_updated: bool = Field(default=False, description="Whether archive was updated")
    commit_created: bool = Field(default=False, description="Whether commit was created")
    commit_info: CommitInfo | None = Field(
        default=None, description="Commit information if created"
    )
    pushed_to_remote: bool = Field(
        default=False, description="Whether changes were pushed to remote"
    )
    files_changed: int = Field(ge=0, default=0, description="Total files changed")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
