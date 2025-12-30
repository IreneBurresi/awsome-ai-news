Feature: News Clustering via LLM
  As a news aggregation pipeline
  I want to cluster similar articles into news topics using Gemini LLM
  So that I can present coherent news stories instead of individual articles

  Background:
    Given the Step 3 configuration is enabled
    And the Gemini API is available

  Scenario: Cluster multiple articles about the same topic
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | OpenAI Releases GPT-5                 | gpt-5-release-abc      |
      | GPT-5: What You Need to Know          | gpt-5-guide-def        |
      | First Look at GPT-5                   | gpt-5-first-look-ghi   |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 1 news cluster should be created
    And the cluster should contain 3 articles
    And the cluster title should mention "GPT-5"
    And the cluster topic should be "model release"

  Scenario: Create singleton clusters for unrelated articles
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | OpenAI Releases GPT-5                 | gpt-5-release-abc      |
      | DeepMind Protein Folding Breakthrough | deepmind-protein-def   |
      | Meta AI Safety Framework              | meta-safety-ghi        |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 3 news clusters should be created
    And all clusters should be singletons
    And each cluster should have 1 article

  Scenario: Mix of multi-article and singleton clusters
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | AI Regulation Announced               | ai-reg-announce-abc    |
      | New AI Regulation: Analysis           | ai-reg-analysis-def    |
      | Tesla Autopilot Update                | tesla-autopilot-ghi    |
      | Quantum Computing Breakthrough        | quantum-comp-jkl       |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And at least 2 news clusters should be created
    And at least 1 cluster should have multiple articles
    And at least 1 cluster should be a singleton

  Scenario: Handle single article input
    Given I have 1 deduplicated article
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 1 news cluster should be created
    And the cluster should be a singleton

  Scenario: Handle empty article list
    Given I have 0 deduplicated articles
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 0 news clusters should be created
    And 0 API calls should be made

  Scenario: Fallback to singleton clusters when API fails
    Given I have 3 deduplicated articles
    And the Gemini API will fail with timeout
    And fallback to singleton is enabled
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 3 news clusters should be created
    And all clusters should be singletons
    And the fallback flag should be true
    And 1 API failure should be recorded

  Scenario: Fail gracefully when API fails and fallback is disabled
    Given I have 3 deduplicated articles
    And the Gemini API will fail with timeout
    And fallback to singleton is disabled
    When I execute Step 3 clustering
    Then Step 3 should fail
    And 0 news clusters should be created
    And 1 API failure should be recorded
    And an error message should be present

  Scenario: Generate news IDs for clusters
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | AI Article 1                          | ai-article-1-abc       |
      | AI Article 2                          | ai-article-2-def       |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And each cluster should have a unique news_id
    And each news_id should start with "news-"
    And each news_id should be 17 characters long

  Scenario: Validate cluster titles and summaries
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | OpenAI GPT-5 Release                  | gpt-5-release-abc      |
      | GPT-5 Analysis                        | gpt-5-analysis-def     |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And each cluster title should be at least 10 characters
    And each cluster summary should be at least 50 characters
    And cluster titles should not exceed 150 characters
    And cluster summaries should not exceed 500 characters

  Scenario: Step 3 disabled
    Given the Step 3 configuration is disabled
    And I have 5 deduplicated articles
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 0 news clusters should be created
    And 0 API calls should be made

  Scenario: High volume clustering
    Given I have 50 deduplicated articles
    When I execute Step 3 clustering
    Then Step 3 should complete in less than 30 seconds
    And Step 3 should succeed
    And at least 1 news cluster should be created
    And the total articles clustered should be 50

  Scenario: Track API statistics
    Given I have 10 deduplicated articles
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And exactly 1 API call should be made
    And 0 API failures should be recorded
    And the statistics should report:
      | metric              | value |
      | total_clusters      | >= 1  |
      | articles_clustered  | 10    |
      | api_calls           | 1     |
      | api_failures        | 0     |

  Scenario: Cluster keywords extraction
    Given I have these deduplicated articles:
      | title                                  | slug                    |
      | Machine Learning Breakthrough         | ml-breakthrough-abc    |
      | Neural Networks Advance               | neural-networks-def    |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And each cluster should have keywords
    And keywords should not exceed 10 per cluster

  Scenario: Missing API key triggers fallback
    Given I have 3 deduplicated articles
    And no API key is configured
    And fallback to singleton is enabled
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And the fallback flag should be true
    And 3 singleton clusters should be created

  Scenario: Preserve article metadata in clusters
    Given I have these deduplicated articles:
      | title                 | slug            |
      | Article A            | article-a-abc  |
      | Article B            | article-b-def  |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And cluster article_slugs should match input slugs
    And cluster article_count should match slugs length

  Scenario: Create coherent news topics
    Given I have these deduplicated articles:
      | title                                     | slug                      |
      | OpenAI Announces GPT-5                   | openai-gpt5-ann-abc      |
      | GPT-5 Features Revealed                  | gpt5-features-def        |
      | Industry Reacts to GPT-5                 | industry-react-gpt5-ghi  |
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And 1 news cluster should be created
    And the cluster should have a coherent topic
    And the cluster summary should synthesize all articles
    And the cluster should identify main topic

  Scenario: Retry on transient API failures
    Given I have 3 deduplicated articles
    And the Gemini API will fail 2 times then succeed
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And exactly 1 successful API call should be recorded
    And news clusters should be created

  Scenario: Validate article count matches slugs
    Given I have 3 deduplicated articles
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And for each cluster article_count should equal slugs length

  Scenario: Singleton cluster fallback validation
    Given I have these deduplicated articles:
      | title     | slug         | content                                                |
      | Short     | short-abc    | Brief                                                  |
      | Long Title Here | long-def | This is a much longer piece of content that exceeds the minimum requirements |
    And the Gemini API will fail
    And fallback to singleton is enabled
    When I execute Step 3 clustering
    Then Step 3 should succeed
    And all singleton titles should be at least 10 characters
    And all singleton summaries should be at least 50 characters
