# LLM Prompts

This directory contains all LLM prompts used in the pipeline, extracted from code into YAML files for easier maintenance and iteration.

## Structure

Each prompt file follows this structure:

```yaml
# Comment with step name and model
system_prompt: |
  The system/role prompt defining the AI's behavior

user_prompt: |
  The user prompt template with {variables} to be filled at runtime
```

## Prompt Files

| File | Step | Model | Purpose |
|------|------|-------|---------|
| `step3_clustering.yaml` | Step 3 | gemini-2.5-flash-lite | Cluster articles into news events |
| `step4_multi_dedup.yaml` | Step 4 | gemini-2.5-flash-lite | Deduplicate news across multiple days |
| `step5_selection.yaml` | Step 5 | gemini-2.5-flash-lite | Categorize and score news |
| `step6_enhancement.yaml` | Step 6 | gemini-2.5-flash-lite | Enhance news with Google Search grounding |

## Usage

Prompts are loaded via `src/utils/prompt_loader.py`:

```python
from src.utils.prompt_loader import get_prompt_loader

loader = get_prompt_loader()
prompt = loader.format_prompt(
    "step3_clustering",
    num_articles=10,
    articles_formatted="...",
    max_clusters=20,
    min_cluster_size=1
)
```

## Variables Reference

### step3_clustering.yaml
- `num_articles` - Number of input articles
- `articles_formatted` - Formatted article list
- `max_clusters` - Maximum clusters to create
- `min_cluster_size` - Minimum articles per cluster

### step4_multi_dedup.yaml
- `num_today_news` - Number of today's news items
- `today_news_formatted` - Formatted today's news
- `num_cached_news` - Number of cached news items
- `cached_news_formatted` - Formatted cached news
- `lookback_days` - Days to look back
- `similarity_threshold` - Threshold for merging (0.0-1.0)

### step5_selection.yaml
- `num_news` - Number of news items
- `news_formatted` - Formatted news list
- `categories_description` - Available categories description
- `recency_weight` - Weight for recency factor
- `source_priority_weight` - Weight for source priority
- `content_quality_weight` - Weight for content quality
- `engagement_weight` - Weight for engagement potential

### step6_enhancement.yaml
- `news_id` - News unique ID
- `title` - News title
- `category` - News category
- `main_topic` - Main topic
- `summary` - News summary
- `keywords` - Comma-separated keywords
- `article_count` - Number of articles in cluster

## Editing Prompts

To modify prompts:

1. Edit the YAML file directly
2. Test changes by running the pipeline
3. No code changes needed - prompts are loaded at runtime

## Best Practices

- Keep `system_prompt` role-focused and general
- Put task-specific instructions in `user_prompt`
- Use clear variable names in `{curly_braces}`
- Document expected format for complex variables
- Version control all prompt changes
