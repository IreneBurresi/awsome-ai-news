Feature: Article Deduplication
  As a news aggregation pipeline
  I want to deduplicate articles based on slug matching
  So that I can avoid showing the same article multiple times

  Background:
    Given the Step 2 configuration is enabled
    And the cache system is ready

  Scenario: Deduplicate with empty cache
    Given I have no cached articles
    And I receive 3 new articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And all 3 articles should be unique
    And 0 duplicates should be found
    And the deduplication rate should be 0%
    And the cache should be updated with 3 articles

  Scenario: All articles are duplicates
    Given I have 5 articles cached from 2 days ago
    And I receive the same 5 articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And 0 articles should be unique
    And 5 duplicates should be found
    And the deduplication rate should be 100%
    And the cache should not be updated

  Scenario: Partial deduplication
    Given I have these cached articles:
      | title                  | slug                    | days_ago |
      | AI Model Released      | ai-model-released-abc   | 3        |
      | ML Breakthrough        | ml-breakthrough-def     | 3        |
    And I receive these articles from Step 1:
      | title                  | slug                    |
      | AI Model Released      | ai-model-released-abc   |
      | New Research Published | new-research-pub-ghi    |
      | Latest AI News         | latest-ai-news-jkl      |
    When I execute Step 2
    Then Step 2 should succeed
    And 2 articles should be unique
    And 1 duplicate should be found
    And the deduplication rate should be 33%
    And the unique articles should be:
      | title                  |
      | New Research Published |
      | Latest AI News         |

  Scenario: Cache rotation after 10 days
    Given I have articles cached from 15 days ago
    And I have articles cached from 5 days ago
    And I receive the same articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And only articles from 5 days ago should be considered
    And articles from 15 days ago should be ignored
    And the cache should be loaded from 1 file

  Scenario: Handle corrupted cache files gracefully
    Given I have 1 valid cache file from 2 days ago
    And I have 1 corrupted cache file from 3 days ago
    And I receive 3 new articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And the corrupted file should be skipped
    And the valid file should be loaded
    And 1 cache file should be marked as corrupted
    And 1 cache file should be loaded successfully

  Scenario: High volume deduplication
    Given I have 500 articles cached from recent days
    And I receive 1000 articles from Step 1
    And 400 of them are duplicates
    When I execute Step 2
    Then Step 2 should complete in less than 5 seconds
    And 600 articles should be unique
    And 400 duplicates should be found
    And the deduplication rate should be 40%

  Scenario: Multiple runs accumulate articles
    Given I have no cached articles
    When I execute Step 2 with 10 articles
    Then 10 articles should be unique
    When I execute Step 2 with 10 different articles
    Then 10 articles should be unique
    And 20 articles should be in cache
    When I execute Step 2 with 5 articles from the first batch
    Then 0 articles should be unique
    And 5 duplicates should be found

  Scenario: Internal duplicate detection
    Given I have no cached articles
    And I receive these articles with duplicate slugs:
      | title            | slug              |
      | Article One      | article-one-abc   |
      | Article One Copy | article-one-abc   |
      | Article Two      | article-two-def   |
    When I execute Step 2
    Then Step 2 should succeed
    And 2 articles should be unique
    And internal duplicates should be prevented
    And only the first occurrence should be kept

  Scenario: Preserve article priority during deduplication
    Given I have cached this article:
      | title       | slug        | feed_priority | days_ago |
      | AI News     | ai-news-abc | 8             | 2        |
    And I receive this article:
      | title       | slug        | feed_priority |
      | AI News     | ai-news-abc | 10            |
    When I execute Step 2
    Then Step 2 should succeed
    And the article should be marked as duplicate
    And the cached version should be preserved

  Scenario: Daily cache file creation
    Given I have no cached articles
    And I receive 5 articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And a cache file should be created for today
    And the cache file should contain 5 articles
    And the cache file name should match today's date

  Scenario: Slug-based exact matching
    Given I have cached this article:
      | title           | slug            | days_ago |
      | Original Title  | article-abc123  | 2        |
    And I receive these articles:
      | title              | slug            |
      | Different Title    | article-abc123  |
      | Similar Title      | article-xyz456  |
    When I execute Step 2
    Then Step 2 should succeed
    And 1 duplicate should be found
    And 1 article should be unique
    And matching should be based on slug only

  Scenario: Handle empty input gracefully
    Given I have 10 articles in cache
    And I receive 0 articles from Step 1
    When I execute Step 2
    Then Step 2 should succeed
    And 0 articles should be unique
    And 0 duplicates should be found
    And the deduplication rate should be 0%
    And the cache should not be updated

  Scenario: Statistics reporting
    Given I have 100 articles cached from recent days
    And I receive 50 articles from Step 1
    And 30 of them are duplicates
    When I execute Step 2
    Then the statistics should report:
      | metric               | value |
      | input_articles       | 50    |
      | cache_articles       | 100   |
      | duplicates_found     | 30    |
      | unique_articles      | 20    |
      | deduplication_rate   | 0.6   |
      | cache_files_loaded   | >=1   |
      | cache_files_corrupted| 0     |

  Scenario: Cache cleanup integration
    Given I have articles cached from these days:
      | days_ago | count |
      | 1        | 10    |
      | 5        | 15    |
      | 9        | 20    |
      | 11       | 25    |
      | 15       | 30    |
    When I execute Step 2 with 5 new articles
    Then only cache files from last 10 days should be loaded
    And 45 articles should be in cache (from days 1, 5, 9)
    And 55 articles should be ignored (from days 11, 15)

  Scenario: Concurrent processing safety
    Given the cache is being accessed concurrently
    And I receive 20 articles from Step 1
    When I execute Step 2
    Then Step 2 should complete successfully
    And the cache should remain consistent
    And no data should be lost

  # Edge Cases

  Scenario: Handle malformed cache file dates
    Given I have a cache file with invalid date format
    When I execute Step 2 with 5 new articles
    Then the malformed file should be skipped
    And the file should be marked as corrupted
    And Step 2 should continue processing

  Scenario: Very old articles in new feed
    Given I have no cached articles
    And I receive an article published 30 days ago
    When I execute Step 2
    Then Step 2 should succeed
    And the article should be processed normally
    And age should not affect deduplication logic

  Scenario: UTF-8 and special characters in slugs
    Given I have cached this article:
      | title                | slug                        | days_ago |
      | AI研究突破           | ai研究突破-abc123           | 2        |
    And I receive this article:
      | title                | slug                        |
      | AI研究突破           | ai研究突破-abc123           |
    When I execute Step 2
    Then Step 2 should succeed
    And the article should be detected as duplicate
    And UTF-8 characters should be handled correctly
