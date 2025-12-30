"""Step 3: News Clustering via Gemini LLM.

This step filters AI-related articles and clusters them into news topics.
Uses structured output with Pydantic for type-safe clustering.
"""

import hashlib
import os
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.articles import ProcessedArticle
from src.models.config import Step3Config
from src.models.news import NewsCluster, Step3Result

# AI-related keywords for filtering (200+ keywords)
AI_KEYWORDS = {
    # Core AI terms
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural net", "llm", "large language model",
    "generative ai", "gen ai", "genai," "ml", "dl", "agi", "artificial general intelligence",
    # AI Companies (50+)
    "openai", "anthropic", "google ai", "deepmind", "google deepmind",
    "meta ai", "microsoft ai", "nvidia", "tesla ai", "apple intelligence",
    "cohere", "stability ai", "midjourney", "runway", "character ai",
    "hugging face", "huggingface", "replicate", "adept", "inflection ai", "aleph alpha",
    "ai21 labs", "scale ai", "databricks", "cerebras", "graphcore",
    "sambanova", "groq", "mistral ai", "together ai", "perplexity",
    "anthropic claude", "openai gpt", "baidu", "alibaba ai", "tencent ai",
    "amazon ai", "aws ai", "salesforce ai", "oracle ai", "ibm watson",
    "lightmatter", "rain ai", "d-matrix", "synthesia", "reka ai",
    "neuralink", "figure ai", "sanctuary ai", "1x technologies",
    "waymo", "cruise", "argo ai", "mobileye", "aurora",
    "anyscale", "modal labs", "weights & biases", "wandb",
    # AI Models and Products (100+)
    "gpt", "gpt-1", "gpt-2", "gpt-3", "gpt-3.5", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
    "chatgpt", "gpt-5", "davinci", "curie", "babbage", "ada",
    "claude", "claude 1", "claude 2", "claude 3", "claude 3.5", "claude opus", "claude sonnet",
    "claude haiku", "claude instant",
    "gemini", "gemini 1.0", "gemini 1.5", "gemini 2.0", "gemini pro", "gemini ultra",
    "gemini flash", "gemini nano", "bard", "palm", "palm 2", "palm-e",
    "llama", "llama 1", "llama 2", "llama 3", "llama 3.1", "llama 3.2", "llama 3.3",
    "code llama", "alpaca", "vicuna", "orca", "guanaco",
    "bert", "roberta", "albert", "electra", "deberta", "t5", "ul2", "flan-t5",
    "gpt-j", "gpt-neox", "bloom", "opt", "falcon", "mpt", "stablelm",
    "mistral", "mixtral", "mistral 7b", "mixtral 8x7b", "mixtral 8x22b",
    "phi", "phi-1", "phi-2", "phi-3", "orca 2",
    "stable diffusion", "sdxl", "sd 1.5", "sd 2.1", "dall-e", "dall-e 2", "dall-e 3", "dalle",
    "midjourney v5", "midjourney v6", "imagen", "imagen 2", "firefly",
    "whisper", "whisper large", "codex", "copilot", "github copilot", "cursor",
    "codeium", "tabnine", "replit ghostwriter",
    "sora", "runway gen-2", "runway gen-3", "pika", "d-id",
    "stable video", "make-a-video", "dreamfusion",
    # Technical Terms (80+)
    "transformer", "attention mechanism", "self-attention", "multi-head attention",
    "cross-attention",
    "encoder", "decoder", "encoder-decoder", "autoencoder", "vae", "variational autoencoder",
    "gan", "generative adversarial network", "diffusion model", "latent diffusion", "denoising",
    "tokenization", "tokenizer", "embedding", "vector embedding", "word embedding",
    "sentence embedding", "contextualized embedding",
    "fine-tuning", "pre-training", "transfer learning", "few-shot learning", "few-shot",
    "zero-shot", "one-shot", "n-shot", "prompt engineering", "prompting", "prompt tuning",
    "in-context learning", "chain of thought", "cot", "reasoning", "step-by-step reasoning",
    "reinforcement learning", "rlhf", "reinforcement learning from human feedback",
    "ppo", "proximal policy optimization", "dpo", "direct preference optimization",
    "reward model", "policy gradient", "q-learning", "actor-critic",
    "backpropagation", "gradient descent", "stochastic gradient descent", "sgd",
    "optimizer", "adam", "adamw", "rmsprop", "momentum",
    "learning rate", "learning rate schedule", "warmup", "hyperparameter", "hyperparameter tuning",
    "neural architecture", "convolutional neural network", "cnn", "convolution",
    "recurrent neural network", "rnn", "lstm", "long short-term memory", "gru",
    "attention head", "feedforward", "mlp", "multilayer perceptron",
    "activation function", "relu", "gelu", "swish", "tanh", "sigmoid",
    "softmax", "layer norm", "batch normalization", "batch norm", "dropout", "regularization",
    "overfitting", "underfitting", "bias-variance tradeoff", "early stopping",
    "loss function", "cross-entropy", "mse", "mae", "accuracy",
    "precision", "recall", "f1 score", "auc", "roc", "confusion matrix",
    "gradient", "weight", "parameter", "checkpoint",
    "epoch", "batch size", "mini-batch", "iteration",
    # AI Application Areas
    "natural language processing", "nlp", "computer vision", "cv",
    "speech recognition", "text-to-speech", "tts", "speech synthesis",
    "image generation", "text-to-image", "image-to-image",
    "video generation", "text-to-video", "audio generation",
    "code generation", "code completion", "program synthesis",
    "robotics", "autonomous vehicles", "self-driving", "autonomous driving",
    "recommendation system", "recommender", "personalization",
    "anomaly detection", "fraud detection", "predictive analytics",
    "sentiment analysis", "named entity recognition", "ner",
    "machine translation", "neural machine translation", "nmt",
    "question answering", "qa system", "conversational ai",
    "chatbot", "virtual assistant", "voice assistant",
    # AI Frameworks and Tools
    "pytorch", "tensorflow", "keras", "jax", "flax",
    "transformers", "langchain", "llama index", "llamaindex",
    "vector database", "vectordb", "pinecone", "weaviate", "qdrant",
    "milvus", "chromadb", "faiss", "annoy", "pgvector",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "ray", "dask", "spark ml", "mlflow", "kubeflow",
    # AI Concepts and Methods
    "supervised learning", "unsupervised learning", "semi-supervised",
    "self-supervised", "contrastive learning", "metric learning",
    "multi-modal", "multimodal", "vision-language", "vlm",
    "foundation model", "base model", "language model", "vision model",
    "mixture of experts", "moe", "sparse model", "dense model",
    "quantization", "pruning", "distillation", "knowledge distillation",
    "model compression", "efficient ai", "edge ai", "tiny ml",
    "federated learning", "distributed training", "model parallelism",
    "data parallelism", "pipeline parallelism",
    # AI Safety and Ethics
    "ai safety", "ai alignment", "alignment research", "red teaming",
    "adversarial examples", "adversarial attack", "robustness",
    "interpretability", "explainability", "xai", "explainable ai",
    "bias", "fairness", "ai ethics", "responsible ai",
    "hallucination", "ai hallucination", "factuality", "truthfulness",
    "jailbreak", "prompt injection", "ai security",
    # AI Infrastructure and Compute
    "gpu", "tpu", "tensor processing unit", "cuda", "compute",
    "inference", "training", "model serving", "deployment",
    "mlops", "llmops", "model monitoring", "a100", "h100",
    # Research and Academic
    "arxiv", "neurips", "icml", "iclr", "cvpr", "acl",
    "research paper", "preprint", "benchmark", "dataset",
    "eval", "evaluation", "ablation study",
    # AI Agents and Tools
    "ai agent", "autonomous agent", "tool use", "function calling",
    "retrieval augmented generation", "rag", "grounding", "search grounding",
    "memory", "long-term memory", "context window", "context length",
    # Emerging AI Concepts
    "world model", "reasoning model", "o1", "chain-of-thought reasoning",
    "test-time compute", "scaling laws", "emergent abilities",
    "learning to learn", "meta-learning",
}


class GeminiClusteringResponse(BaseModel):
    """Structured response from Gemini for clustering."""

    clusters: list[NewsCluster] = Field(description="List of identified news clusters")
    total_articles_processed: int = Field(ge=0, description="Total articles processed")
    clustering_rationale: str = Field(description="Brief explanation of clustering logic used")


def _is_ai_related(article: ProcessedArticle) -> bool:
    """
    Check if article is AI-related based on keywords.

    Args:
        article: Article to check

    Returns:
        True if article mentions AI-related keywords
    """
    # Combine title and content for checking
    text = f"{article.title} {article.content or ''}".lower()

    # Check if any AI keyword appears in the text
    return any(keyword in text for keyword in AI_KEYWORDS)


def _generate_news_id(title: str, article_slugs: list[str]) -> str:
    """
    Generate unique news ID from title and article slugs.

    Args:
        title: News title
        article_slugs: List of article slugs in cluster

    Returns:
        Unique news ID (format: news-{hash[:12]})
    """
    # Combine title and sorted slugs for deterministic ID
    content = f"{title}:{'|'.join(sorted(article_slugs))}"
    hash_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"news-{hash_hex}"


def _prepare_articles_for_prompt(articles: list[ProcessedArticle]) -> list[dict]:
    """
    Prepare articles data for LLM prompt.

    Args:
        articles: List of processed articles

    Returns:
        List of article dictionaries with title, url, content preview
    """
    prepared = []
    for article in articles:
        # Extract first 200 chars of content as preview
        content_preview = ""
        if article.content:
            content_preview = article.content[:200].strip()

        prepared.append(
            {
                "slug": article.slug,
                "title": article.title,
                "url": str(article.url),
                "content_preview": content_preview,
                "feed": article.feed_name,
                "published": article.published_date.isoformat()
                if article.published_date
                else "unknown",
            }
        )

    return prepared


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_gemini_clustering(
    articles_data: list[dict],
    config: Step3Config,
    api_key: str,
) -> GeminiClusteringResponse:
    """
    Call Gemini API for clustering with structured output.

    Args:
        articles_data: Prepared articles data
        config: Step 3 configuration
        api_key: Gemini API key

    Returns:
        Structured clustering response

    Raises:
        Exception: On API failures after retries
    """
    from google import genai

    # Create client
    client = genai.Client(api_key=api_key)

    # Build prompt
    prompt = f"""You are an AI news clustering system.
Your task is to group similar AI news articles into coherent news topics.

**Input Articles** ({len(articles_data)} articles):
{_format_articles_for_prompt(articles_data)}

**Clustering Instructions**:
1. Group articles that discuss the SAME news event or topic
2. Articles about different aspects of the same story should be in one cluster
3. Maximum {config.max_clusters} clusters
4. Minimum {config.min_cluster_size} article(s) per cluster (but singletons are OK)
5. Each cluster needs:
   - A clear, concise title (10-150 chars)
   - A comprehensive summary (50-500 chars) covering all articles
   - The main topic category (e.g., "model release", "research breakthrough", "policy")
   - Keywords (max 10)
   - List of article slugs included

**Output Format**: JSON with clusters array

**Important**:
- Be precise: only group truly related articles
- Prefer specific clusters over generic grouping
- If articles don't fit existing clusters, create new ones
- Include ALL {len(articles_data)} articles in the output (no article left behind)"""

    logger.debug("Calling Gemini API for clustering", model=config.llm_model)

    # Make API call with structured output (using Pydantic class directly)
    response = client.models.generate_content(
        model=config.llm_model,
        contents=prompt,
        config={
            "temperature": config.temperature,
            "response_mime_type": "application/json",
            "response_schema": GeminiClusteringResponse,
        },
    )

    logger.debug("Gemini API response received", response_text=response.text[:200])  # type: ignore

    # Parse and validate with Pydantic
    clustering_response = GeminiClusteringResponse.model_validate_json(response.text)  # type: ignore

    return clustering_response


def _format_articles_for_prompt(articles_data: list[dict]) -> str:
    """Format articles for inclusion in prompt."""
    lines = []
    for i, article in enumerate(articles_data, 1):
        lines.append(
            f"{i}. [{article['slug']}] {article['title']}\n"
            f"   Source: {article['feed']} | Published: {article['published']}\n"
            f"   Preview: {article['content_preview']}\n"
        )
    return "\n".join(lines)


def _create_singleton_clusters(articles: list[ProcessedArticle]) -> list[NewsCluster]:
    """
    Create singleton clusters (fallback when Gemini fails).

    Each article becomes its own cluster.

    Args:
        articles: List of processed articles

    Returns:
        List of singleton news clusters
    """
    logger.warning("Creating singleton clusters as fallback")
    clusters = []

    for article in articles:
        news_id = _generate_news_id(article.title, [article.slug])

        # Use article title as news title, ensuring min length 10 chars
        title = article.title[:150]
        if len(title) < 10:
            title = f"News: {article.title}"[:150]

        # Use content preview as summary, or generate from title
        if article.content and len(article.content) >= 50:
            summary = article.content[:500]
        else:
            # Generate summary ensuring min 50 chars
            summary = (
                f"News about {article.title}. Published by {article.feed_name}. "
                f"This article discusses {article.title.lower()}."
            )
            summary = summary[:500]  # Cap at 500 chars

        # Extract some keywords from title
        words = article.title.lower().split()
        keywords = [w for w in words if len(w) > 4][:5]  # Take first 5 long words

        cluster = NewsCluster(
            news_id=news_id,
            title=title,
            summary=summary,
            article_slugs=[article.slug],
            article_count=1,
            main_topic="singleton",
            keywords=keywords,
            created_at=datetime.utcnow(),
        )
        clusters.append(cluster)

    return clusters


async def run_step3(
    config: Step3Config,
    articles: list[ProcessedArticle],
    api_key: str | None = None,
) -> Step3Result:
    """
    Execute Step 3: News Clustering.

    Groups deduplicated articles into news topics using Gemini LLM.

    Args:
        config: Step 3 configuration
        articles: List of deduplicated articles from Step 2
        api_key: Optional Gemini API key (defaults to GEMINI_API_KEY env var)

    Returns:
        Step3Result with news clusters and statistics
    """
    logger.info("Starting Step 3: News clustering", num_articles=len(articles))

    if not config.enabled:
        logger.info("Step 3 is disabled, skipping")
        return Step3Result(
            success=True,
            news_clusters=[],
            total_clusters=0,
            singleton_clusters=0,
            multi_article_clusters=0,
            articles_clustered=0,
            api_calls=0,
        )

    if len(articles) == 0:
        logger.info("No articles to cluster, skipping Step 3")
        return Step3Result(
            success=True,
            news_clusters=[],
            total_clusters=0,
            singleton_clusters=0,
            multi_article_clusters=0,
            articles_clustered=0,
            api_calls=0,
        )

    # Filter AI-related articles only
    ai_articles = [article for article in articles if _is_ai_related(article)]
    filtered_count = len(articles) - len(ai_articles)

    logger.info(
        "Filtered articles",
        total_input=len(articles),
        ai_related=len(ai_articles),
        filtered_out=filtered_count,
    )

    if len(ai_articles) == 0:
        logger.info("No AI-related articles to cluster, skipping Step 3")
        return Step3Result(
            success=True,
            news_clusters=[],
            total_clusters=0,
            singleton_clusters=0,
            multi_article_clusters=0,
            articles_clustered=0,
            api_calls=0,
        )

    # Use ai_articles for clustering instead of all articles
    articles = ai_articles

    errors = []
    api_calls = 0
    api_failures = 0
    fallback_used = False

    try:
        # Get API key
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            error_msg = "GEMINI_API_KEY not provided and not in environment"
            logger.error(error_msg)
            errors.append(error_msg)

            # Use fallback
            if config.fallback_to_singleton:
                clusters = _create_singleton_clusters(articles)
                fallback_used = True
            else:
                return Step3Result(
                    success=False,
                    errors=errors,
                    total_clusters=0,
                    singleton_clusters=0,
                    multi_article_clusters=0,
                    articles_clustered=0,
                    api_calls=0,
                    api_failures=1,
                )
        else:
            # Prepare articles for prompt
            articles_data = _prepare_articles_for_prompt(articles)

            logger.info("Calling Gemini API for clustering")

            try:
                # Make API call with retries
                response = await _call_gemini_clustering(articles_data, config, key)
                api_calls += 1

                # Extract clusters from response
                clusters = response.clusters

                # Generate news_id for each cluster if not present
                for cluster in clusters:
                    if not cluster.news_id or cluster.news_id == "":
                        cluster.news_id = _generate_news_id(cluster.title, cluster.article_slugs)

                logger.info(
                    "Clustering completed",
                    total_clusters=len(clusters),
                    articles_processed=response.total_articles_processed,
                )

            except Exception as e:
                error_msg = f"Gemini API call failed: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                api_failures += 1

                # Use fallback if configured
                if config.fallback_to_singleton:
                    logger.warning("Using singleton fallback due to API failure")
                    clusters = _create_singleton_clusters(articles)
                    fallback_used = True
                else:
                    return Step3Result(
                        success=False,
                        errors=errors,
                        total_clusters=0,
                        singleton_clusters=0,
                        multi_article_clusters=0,
                        articles_clustered=0,
                        api_calls=api_calls,
                        api_failures=api_failures,
                    )

        # Calculate statistics
        singleton_count = sum(1 for c in clusters if c.article_count == 1)
        multi_count = len(clusters) - singleton_count
        total_articles = sum(c.article_count for c in clusters)

        logger.info(
            "Step 3 completed",
            total_clusters=len(clusters),
            singletons=singleton_count,
            multi_article=multi_count,
            articles_clustered=total_articles,
            fallback_used=fallback_used,
        )

        return Step3Result(
            success=True,
            news_clusters=clusters,
            total_clusters=len(clusters),
            singleton_clusters=singleton_count,
            multi_article_clusters=multi_count,
            articles_clustered=total_articles,
            api_calls=api_calls,
            api_failures=api_failures,
            errors=errors,
            fallback_used=fallback_used,
        )

    except Exception as e:
        error_msg = f"Step 3 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)

        return Step3Result(
            success=False,
            errors=errors,
            total_clusters=0,
            singleton_clusters=0,
            multi_article_clusters=0,
            articles_clustered=0,
            api_calls=api_calls,
            api_failures=api_failures + 1,
            fallback_used=fallback_used,
        )
