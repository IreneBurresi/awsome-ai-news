"""RSS feed data models for Step 8."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class RSSItem(BaseModel):
    """Single RSS item."""

    title: str = Field(min_length=5, max_length=200, description="Item title")
    description: str = Field(min_length=50, max_length=4000, description="Item description")
    link: HttpUrl = Field(description="Item link URL")
    pub_date: datetime = Field(description="Publication date")
    guid: str = Field(description="Unique GUID (news_id)")
    categories: list[str] = Field(default_factory=list, max_length=3, description="Item categories")


class RSSFeed(BaseModel):
    """Complete RSS feed."""

    title: str = Field(description="Feed title")
    description: str = Field(description="Feed description")
    link: HttpUrl = Field(description="Feed link URL")
    language: str = Field(default="en-US", description="Feed language")
    pub_date: datetime = Field(description="Feed publication date")
    items: list[RSSItem] = Field(description="Feed items")


class Step8Result(BaseModel):
    """Result from Step 8: RSS feed generation."""

    success: bool = Field(description="Whether step completed successfully")
    daily_feed_path: Path | None = Field(default=None, description="Path to daily feed file")
    weekly_feed_path: Path | None = Field(default=None, description="Path to weekly feed file")
    daily_items_count: int = Field(ge=0, default=0, description="Number of daily items")
    weekly_items_count: int = Field(ge=0, default=0, description="Number of weekly items")
    feeds_valid: bool = Field(default=False, description="Whether feeds pass RSS 2.0 validation")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
