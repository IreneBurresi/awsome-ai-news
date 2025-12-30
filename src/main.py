#!/usr/bin/env python3
"""Main entry point for the awesome-ai-news pipeline.

This orchestrates the execution of all 8 steps:
- Step 0: Cache management and cleanup
- Step 1: RSS feed ingestion
- Step 2: Article deduplication
- Step 3: News clustering
- Step 4: Multi-day deduplication
- Step 5: Top news selection
- Step 6: Content enhancement
- Step 7: Repository update
- Step 8: RSS generation

Usage:
    python -m src.main
    python -m src.main --dry-run
    python -m src.main --config config/pipeline.yaml
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from loguru import logger

from src.models.config import (
    FeedsConfig,
    PipelineConfig,
    Step0Config,
    Step1Config,
    Step2Config,
    Step3Config,
    Step4Config,
    Step5Config,
    Step6Config,
    Step7Config,
    Step8Config,
)
from src.steps.step0_cache import run_step0
from src.steps.step1_ingestion import run_step1
from src.steps.step2_dedup import run_step2
from src.steps.step3_clustering import run_step3
from src.steps.step4_multi_dedup import run_step4
from src.steps.step5_selection import run_step5
from src.steps.step6_enhancement import run_step6
from src.steps.step7_repo import run_step7
from src.steps.step8_rss import run_step8
from src.utils.cache import CacheManager
from src.utils.config_loader import load_feeds_config, load_pipeline_config

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(self, message: str, step: str, recoverable: bool = False):
        self.step = step
        self.recoverable = recoverable
        super().__init__(message)


class CriticalError(PipelineError):
    """Critical error that stops the pipeline."""

    def __init__(self, message: str, step: str):
        super().__init__(message, step, recoverable=False)


class RecoverableError(PipelineError):
    """Recoverable error that allows pipeline to continue."""

    def __init__(self, message: str, step: str):
        super().__init__(message, step, recoverable=True)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_stats(label: str, value: Any) -> None:
    """Print a formatted stat line."""
    print(f"  • {label}: {value}")


def print_summary(results: dict[str, Any], elapsed_time: float) -> None:
    """Print final pipeline summary."""
    print_header("📈 Pipeline Summary")

    if "step0" in results:
        result0 = results["step0"]
        print("Step 0 (Cache Management):")
        print_stats("Status", "✅ Success" if result0.success else "❌ Failed")
        print_stats("Entries cleaned", result0.cache_cleaned)
        print_stats("Cache backed up", result0.cache_backed_up)

    if "step1" in results:
        result1 = results["step1"]
        print("\nStep 1 (RSS Ingestion):")
        print_stats("Status", "✅ Success" if result1.success else "❌ Failed")
        total_feeds = result1.feeds_fetched + result1.feeds_failed
        print_stats("Feeds fetched", f"{result1.feeds_fetched}/{total_feeds}")
        print_stats("Articles fetched", len(result1.articles))
        if result1.slug_collisions > 0:
            print_stats("Slug collisions", result1.slug_collisions)

    if "step2" in results:
        result2 = results["step2"]
        print("\nStep 2 (Deduplication):")
        print_stats("Status", "✅ Success" if result2.success else "❌ Failed")
        print_stats("Unique articles", len(result2.unique_articles))
        print_stats("Duplicates removed", result2.stats.duplicates_found)
        print_stats("Deduplication rate", f"{result2.stats.deduplication_rate:.1%}")

    if "step3" in results:
        result3 = results["step3"]
        print("\nStep 3 (Clustering):")
        print_stats("Status", "✅ Success" if result3.success else "❌ Failed")
        print_stats("News clusters", result3.total_clusters)
        print_stats("Multi-article clusters", result3.multi_article_clusters)
        print_stats("Singleton clusters", result3.singleton_clusters)
        if result3.fallback_used:
            print_stats("⚠️  Fallback used", "Yes (API failed)")
        if result3.api_calls > 0:
            print_stats("API calls", result3.api_calls)

    if "step4" in results:
        result4 = results["step4"]
        print("\nStep 4 (Multi-day Deduplication):")
        print_stats("Status", "✅ Success" if result4.success else "❌ Failed")
        print_stats("News before dedup", result4.news_before_dedup)
        print_stats("Duplicates found", result4.duplicates_found)
        print_stats("News merged", result4.news_merged)
        print_stats("Unique news", result4.news_after_dedup)
        if result4.fallback_used:
            print_stats("⚠️  Fallback used", "Yes (no merge)")
        if result4.api_calls > 0:
            print_stats("API calls", result4.api_calls)

    if "step5" in results:
        result5 = results["step5"]
        print("\nStep 5 (Top News Selection):")
        print_stats("Status", "✅ Success" if result5.success else "❌ Failed")
        print_stats("Total news", len(result5.all_categorized_news))
        print_stats("Top news selected", len(result5.top_news))
        print_stats("Categories", len(result5.categories_distribution))
        if result5.api_calls > 0:
            print_stats("API calls", result5.api_calls)

    if "step6" in results:
        result6 = results["step6"]
        print("\nStep 6 (Content Enhancement):")
        print_stats("Status", "✅ Success" if result6.success else "❌ Failed")
        print_stats("Enhanced news", len(result6.enhanced_news))
        print_stats("Total external links", result6.total_external_links)
        print_stats("Avg links per news", f"{result6.avg_links_per_news:.1f}")
        if result6.enhancement_failures > 0:
            print_stats("⚠️  Enhancement failures", result6.enhancement_failures)
        if result6.api_calls > 0:
            print_stats("API calls", result6.api_calls)

    if "step7" in results:
        result7 = results["step7"]
        print("\nStep 7 (Repository Update):")
        print_stats("Status", "✅ Success" if result7.success else "❌ Failed")
        print_stats("README updated", "✅" if result7.readme_updated else "❌")
        print_stats("Archive updated", "✅" if result7.archive_updated else "❌")
        print_stats("Commit created", "✅" if result7.commit_created else "❌")
        print_stats("Pushed to remote", "✅" if result7.pushed_to_remote else "❌")
        print_stats("Files changed", result7.files_changed)
        if result7.commit_info:
            print_stats("Commit hash", result7.commit_info.commit_hash)

    if "step8" in results:
        result8 = results["step8"]
        print("\nStep 8 (RSS Feed Generation):")
        print_stats("Status", "✅ Success" if result8.success else "❌ Failed")
        print_stats("Daily items", result8.daily_items_count)
        print_stats("Weekly items", result8.weekly_items_count)
        print_stats("Feeds valid", "✅" if result8.feeds_valid else "❌")

    print("\n" + "=" * 80)
    print(f"⏱️  Total execution time: {elapsed_time:.2f}s")
    print("=" * 80)


async def run_pipeline(
    config: PipelineConfig,
    feeds_config: FeedsConfig,
    cache_manager: CacheManager,
    dry_run: bool = False,
) -> int:
    """
    Execute the complete pipeline.

    Args:
        config: Pipeline configuration
        feeds_config: RSS feeds configuration
        cache_manager: Cache manager instance
        dry_run: If True, skip side effects (commits, pushes)

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    start_time = datetime.now()
    results: dict[str, Any] = {}

    try:
        # ========================================================================
        # STEP 0: Cache Management
        # ========================================================================
        print_header("📦 STEP 0: Cache Management & Cleanup")

        result0 = await run_step0(
            config.step0_cache,
            cache_manager,
            api_key=os.getenv("GEMINI_API_KEY"),
            check_gemini=False,  # Skip health check - will be checked in Step 3
        )
        results["step0"] = result0

        print_stats("Success", result0.success)
        print_stats("Cache entries cleaned", result0.cache_cleaned)
        print_stats("Cache backed up", result0.cache_backed_up)

        if result0.errors:
            print("\n⚠️  Warnings:")
            for error in result0.errors:
                print(f"  • {error}")

        if not result0.success:
            print("\n❌ Step 0 failed!")
            raise CriticalError("Cache management failed", step="Step 0")

        print("\n✅ Step 0 completed successfully!")

        # ========================================================================
        # STEP 1: RSS Feed Ingestion
        # ========================================================================
        print_header("📰 STEP 1: RSS Feed Ingestion")

        print(f"Fetching from {len(feeds_config.feeds)} feeds...")

        result1 = await run_step1(config.step1_ingestion, feeds_config, cache_manager)
        results["step1"] = result1

        print("\n📊 Step 1 Results:")
        print_stats("Success", result1.success)
        print_stats("Feeds fetched", result1.feeds_fetched)
        print_stats("Feeds failed", result1.feeds_failed)
        print_stats("Total articles (raw)", result1.total_articles_raw)
        print_stats("After filtering", result1.articles_after_filter)
        print_stats("Final articles", len(result1.articles))

        if result1.errors:
            print("\n⚠️  Errors during ingestion:")
            for error in result1.errors[:5]:  # Show max 5 errors
                print(f"  • {error}")

        if not result1.success:
            print("\n❌ Step 1 failed!")
            raise CriticalError("RSS ingestion failed", step="Step 1")

        if len(result1.articles) == 0:
            print("\n❌ No articles fetched - this is a critical failure!")
            print("  Possible causes:")
            print("  • Network issues")
            print("  • All RSS feeds are down")
            print("  • Filter keywords too restrictive")
            raise CriticalError("No articles fetched from any feed", step="Step 1")

        print(f"\n✅ Step 1 completed successfully! Fetched {len(result1.articles)} articles")

        # ========================================================================
        # STEP 2: Article Deduplication
        # ========================================================================
        print_header("🔍 STEP 2: Article Deduplication")

        print(f"Deduplicating {len(result1.articles)} articles...")
        print("  • Lookback window: 10 days")
        print("  • Method: Exact slug matching")

        result2 = await run_step2(config.step2_dedup, result1.articles, cache_manager)
        results["step2"] = result2

        print("\n📊 Step 2 Results:")
        print_stats("Success", result2.success)
        print_stats("Input articles", result2.stats.input_articles)
        print_stats("Cached articles", result2.stats.cache_articles)
        print_stats("Duplicates found", result2.stats.duplicates_found)
        print_stats("Unique articles", result2.stats.unique_articles)
        print_stats("Deduplication rate", f"{result2.stats.deduplication_rate:.1%}")

        if result2.errors:
            print("\n⚠️  Errors during deduplication:")
            for error in result2.errors:
                print(f"  • {error}")

        if not result2.success:
            print("\n❌ Step 2 failed!")
            raise CriticalError("Deduplication failed", step="Step 2")

        print(f"\n✅ Step 2 completed successfully! {len(result2.unique_articles)} unique articles")

        # ========================================================================
        # STEP 3: News Clustering
        # ========================================================================
        print_header("🔮 STEP 3: News Clustering via Gemini")

        print(f"Clustering {len(result2.unique_articles)} articles...")
        print(f"  • Model: {config.step3_clustering.llm_model}")
        print(f"  • Max clusters: {config.step3_clustering.max_clusters}")
        print(f"  • Fallback enabled: {config.step3_clustering.fallback_to_singleton}")

        result3 = await run_step3(
            config.step3_clustering,
            result2.unique_articles,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        results["step3"] = result3

        print("\n📊 Step 3 Results:")
        print_stats("Success", result3.success)
        print_stats("News clusters", result3.total_clusters)
        print_stats("Multi-article clusters", result3.multi_article_clusters)
        print_stats("Singleton clusters", result3.singleton_clusters)
        print_stats("Articles clustered", result3.articles_clustered)
        print_stats("API calls", result3.api_calls)
        if result3.fallback_used:
            print_stats("⚠️  Fallback used", "Yes (singleton clusters)")

        if result3.errors:
            print("\n⚠️  Errors during clustering:")
            for error in result3.errors:
                print(f"  • {error}")

        if not result3.success:
            print("\n❌ Step 3 failed!")
            raise CriticalError("Clustering failed", step="Step 3")

        print(f"\n✅ Step 3 completed successfully! {result3.total_clusters} news clusters created")

        # Show sample clusters
        if result3.news_clusters:
            print("\n📰 Sample News Clusters:")
            for i, cluster in enumerate(result3.news_clusters[:3], 1):
                print(f"\n  {i}. {cluster.title}")
                print(f"     Articles: {cluster.article_count}")
                print(f"     Topic: {cluster.main_topic}")
                print(f"     Keywords: {', '.join(cluster.keywords[:5])}")

        # ========================================================================
        # STEP 4: Multi-day News Deduplication
        # ========================================================================
        print_header("🔄 STEP 4: Multi-day News Deduplication")

        print(f"Deduplicating {result3.total_clusters} news clusters...")
        print(f"  • Model: {config.step4_multi_dedup.llm_model}")
        print(f"  • Lookback: {config.step4_multi_dedup.lookback_days} days")
        print(f"  • Similarity threshold: {config.step4_multi_dedup.similarity_threshold}")
        print(f"  • Fallback enabled: {config.step4_multi_dedup.fallback_to_no_merge}")

        result4 = await run_step4(
            config.step4_multi_dedup,
            result3.news_clusters,
            cache_manager,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        results["step4"] = result4

        print("\n📊 Step 4 Results:")
        print_stats("Success", result4.success)
        print_stats("News before dedup", result4.news_before_dedup)
        print_stats("Duplicates found", result4.duplicates_found)
        print_stats("News merged", result4.news_merged)
        print_stats("Unique news", result4.news_after_dedup)
        print_stats("API calls", result4.api_calls)
        if result4.fallback_used:
            print_stats("⚠️  Fallback used", "Yes (no merge)")

        if result4.errors:
            print("\n⚠️  Errors during deduplication:")
            for error in result4.errors:
                print(f"  • {error}")

        if not result4.success:
            print("\n❌ Step 4 failed!")
            raise CriticalError("Multi-day deduplication failed", step="Step 4")

        unique_count = result4.news_after_dedup
        print(f"\n✅ Step 4 completed successfully! {unique_count} unique news clusters")

        # ========================================================================
        # STEP 5: Top News Selection and Categorization
        # ========================================================================
        print_header("🎯 STEP 5: Top News Selection and Categorization")

        print(f"Categorizing and selecting top {config.step5_selection.target_count} news...")
        print(f"  • Target count: {config.step5_selection.target_count}")
        print(f"  • Min quality score: {config.step5_selection.min_quality_score}")

        result5 = await run_step5(
            config.step5_selection,
            result4.unique_news,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        results["step5"] = result5

        print("\n📊 Step 5 Results:")
        print_stats("Success", result5.success)
        print_stats("Total news categorized", len(result5.all_categorized_news))
        print_stats("Top news selected", len(result5.top_news))
        print_stats("API calls", result5.api_calls)

        if result5.categories_distribution:
            print("\n📂 Category Distribution:")
            for category, count in sorted(
                result5.categories_distribution.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  • {category.value}: {count}")

        if result5.errors:
            print("\n⚠️  Errors during selection:")
            for error in result5.errors:
                print(f"  • {error}")

        if not result5.success:
            print("\n❌ Step 5 failed!")
            raise CriticalError("Top news selection failed", step="Step 5")

        print(f"\n✅ Step 5 completed successfully! Selected {len(result5.top_news)} top news")

        if result5.top_news:
            print("\n🏆 Top 3 News (by importance score):")
            for i, cat_news in enumerate(result5.top_news[:3], 1):
                print(f"\n  {i}. {cat_news.news_cluster.title}")
                score = cat_news.importance_score
                category = cat_news.category.value
                print(f"     Score: {score:.1f}/10 | Category: {category}")
                print(f"     Articles: {cat_news.news_cluster.article_count}")

        # ========================================================================
        # STEP 6: Content Enhancement with Web Grounding
        # ========================================================================
        print_header("🌐 STEP 6: Content Enhancement with Grounding")

        print(f"Enhancing {len(result5.top_news)} news with web grounding...")
        print(f"  • Model: {config.step6_enhancement.llm_model}")
        print("  • Grounding: Google Search enabled")

        result6 = await run_step6(
            config.step6_enhancement,
            result5.top_news,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        results["step6"] = result6

        print("\n📊 Step 6 Results:")
        print_stats("Success", result6.success)
        print_stats("Enhanced news", len(result6.enhanced_news))
        print_stats("Total external links", result6.total_external_links)
        print_stats("Avg links per news", f"{result6.avg_links_per_news:.1f}")
        print_stats("API calls", result6.api_calls)

        if result6.errors:
            print("\n⚠️  Errors during enhancement:")
            for error in result6.errors:
                print(f"  • {error}")

        if not result6.success:
            print("\n❌ Step 6 failed!")
            raise CriticalError("Content enhancement failed", step="Step 6")

        print(f"\n✅ Step 6 completed successfully! Enhanced {len(result6.enhanced_news)} news")

        if result6.enhanced_news:
            print("\n📰 Sample Enhanced News:")
            sample = result6.enhanced_news[0]
            print(f"\n  Title: {sample.news.news_cluster.title}")
            print(f"  Summary length: {len(sample.extended_summary)} chars")
            print(f"  External links: {len(sample.external_links)}")
            print(f"  Key points: {len(sample.key_points)}")
            print(f"  Citations: {len(sample.citations)}")
            print(f"  Grounded: {sample.grounded}")

        # ========================================================================
        # STEP 7: Repository Update
        # ========================================================================
        print_header("📝 STEP 7: Repository Update")

        print(f"Updating repository with {len(result6.enhanced_news)} news...")
        print(f"  • README: {config.step7_repo.output_file}")
        print(f"  • Archive: {'Enabled' if config.step7_repo.archive_enabled else 'Disabled'}")
        push_enabled = config.step7_repo.git_push and not dry_run
        print(f"  • Git push: {'Enabled' if push_enabled else 'Disabled'}")

        result7 = await run_step7(
            config.step7_repo,
            result6.enhanced_news,
            dry_run=dry_run,
        )
        results["step7"] = result7

        print("\n📊 Step 7 Results:")
        print_stats("Success", result7.success)
        print_stats("README updated", result7.readme_updated)
        if result7.news_file_created:
            print_stats("News file", str(result7.news_file_created))
        print_stats("Archive updated", result7.archive_updated)
        print_stats("Commit created", result7.commit_created)
        if result7.commit_info:
            print_stats("Commit hash", result7.commit_info.commit_hash)
        print_stats("Pushed to remote", result7.pushed_to_remote)
        print_stats("Files changed", result7.files_changed)

        if not result7.success:
            logger.warning(f"Step 7 completed with errors: {result7.errors}")
            # Continue execution - repo update failures are not critical

        print("\n✅ Step 7 completed!")

        # ========================================================================
        # STEP 8: RSS Feed Generation
        # ========================================================================
        print_header("📡 STEP 8: RSS Feed Generation")

        print(f"Generating RSS feeds from {len(result6.enhanced_news)} news...")
        print(f"  • Daily feed: {config.step8_rss.output_file}")
        print("  • Weekly feed: weekly.xml")

        result8 = await run_step8(
            config.step8_rss,
            result6.enhanced_news,
        )
        results["step8"] = result8

        print("\n📊 Step 8 Results:")
        print_stats("Success", result8.success)
        if result8.daily_feed_path:
            print_stats("Daily feed", str(result8.daily_feed_path))
        print_stats("Daily items", result8.daily_items_count)
        if result8.weekly_feed_path:
            print_stats("Weekly feed", str(result8.weekly_feed_path))
        print_stats("Weekly items", result8.weekly_items_count)
        print_stats("Feeds valid", "✅" if result8.feeds_valid else "❌")

        if not result8.success:
            logger.warning(f"Step 8 completed with errors: {result8.errors}")

        print("\n✅ Step 8 completed!")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        print_summary(results, elapsed)

        if dry_run:
            print("\n🔍 DRY RUN MODE - No changes committed")

        print("\n✅ Pipeline completed successfully!")
        return 0

    except CriticalError as e:
        logger.error(f"Critical error in {e.step}: {e}")
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n❌ Pipeline failed in {e.step}")
        print(f"   Error: {e}")
        print(f"   Elapsed time: {elapsed:.2f}s")

        return 1

    except Exception as e:
        logger.error("Unexpected pipeline error", exc_info=True)
        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n❌ Pipeline failed with unexpected error")
        print(f"   Error: {e}")
        print(f"   Elapsed time: {elapsed:.2f}s")

        return 1


@app.command()
def main(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to pipeline configuration file",
        ),
    ] = Path("config/pipeline.yaml"),
    feeds_file: Annotated[
        Path,
        typer.Option(
            "--feeds",
            "-f",
            help="Path to feeds configuration file",
        ),
    ] = Path("config/feeds.yaml"),
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Path to cache directory"),
    ] = Path("cache"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run without side effects (no commits/pushes)"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """
    Run the awesome-ai-news pipeline.

    Executes all steps: cache management, RSS ingestion, deduplication,
    clustering, selection, enhancement, repository update, and RSS generation.
    """

    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()  # Remove default handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)

    # Add file logging
    log_file = Path("logs/pipeline.log")
    log_file.parent.mkdir(exist_ok=True)
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    logger.add(
        log_file,
        rotation="500 MB",
        retention="30 days",
        compression="zip",
        format=file_format,
        level=log_level,
    )

    print_header("🚀 Awesome AI News Pipeline")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {config_file}")
    print(f"  Feeds: {feeds_file}")
    print(f"  Cache: {cache_dir}")
    print(f"  Dry run: {dry_run}")
    print(f"  Log level: {log_level}")

    try:
        # Load configurations
        logger.info("Loading configuration files")

        if config_file.exists():
            pipeline_config = load_pipeline_config(config_file)
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            # Create default config using load_pipeline_config with empty dict
            from src.models.config import (
                ErrorHandlingConfig,
                LoggingConfig,
                PipelineMetadata,
            )

            pipeline_config = PipelineConfig(
                pipeline=PipelineMetadata(
                    name="awesome-ai-news",
                    version="1.0.0",
                    execution_mode="production",
                ),
                step0_cache=Step0Config(
                    enabled=True,
                    retention={"articles_days": 10, "news_days": 3},
                    backup_on_error=True,
                    cleanup_on_start=True,
                ),
                step1_ingestion=Step1Config(enabled=True),
                step2_dedup=Step2Config(enabled=True),
                step3_clustering=Step3Config(enabled=True),
                step4_multi_dedup=Step4Config(enabled=True),
                step5_selection=Step5Config(enabled=True),
                step6_enhancement=Step6Config(enabled=True),
                step7_repo=Step7Config(enabled=True),
                step8_rss=Step8Config(enabled=True),
                logging=LoggingConfig(),
                error_handling=ErrorHandlingConfig(),
            )

        if feeds_file.exists():
            feeds_config = load_feeds_config(feeds_file)
        else:
            logger.error(f"Feeds file {feeds_file} not found")
            print(f"\n❌ Feeds configuration file not found: {feeds_file}")
            sys.exit(1)

        # Create cache manager
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_manager = CacheManager(cache_dir=cache_dir)

        # Run pipeline
        exit_code = asyncio.run(run_pipeline(pipeline_config, feeds_config, cache_manager, dry_run))

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        logger.error("Failed to start pipeline", exc_info=True)
        print(f"\n❌ Failed to start pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
