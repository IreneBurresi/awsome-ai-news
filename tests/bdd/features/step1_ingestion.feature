Feature: RSS Feed Ingestion
  As a news aggregation pipeline
  I want to fetch and process RSS feeds
  So that I can collect AI news articles for clustering

  Background:
    Given the Step 1 configuration is enabled
    And the cache system is ready

  Scenario: Fetch specialized feed without filtering
    Given a specialized feed "AI Research" with URL "https://example.com/ai-feed.rss"
    And the feed contains 5 articles
    When I execute Step 1
    Then Step 1 should succeed
    And all 5 articles should be accepted
    And each article should have a unique slug
    And the articles should be sorted by date descending

  Scenario: Fetch generalist feed with whitelist filtering
    Given a generalist feed "TechNews" with URL "https://example.com/tech-feed.rss"
    And the feed has whitelist keywords "AI,machine learning,LLM"
    And the feed contains these articles:
      | title                    | should_match |
      | New AI Model Released    | yes          |
      | Latest Smartphone Review | no           |
      | LLM Breakthrough         | yes          |
      | Gaming News Today        | no           |
    When I execute Step 1
    Then Step 1 should succeed
    And 2 articles should be accepted
    And 2 articles should be filtered out

  Scenario: Fetch generalist feed with blacklist filtering
    Given a generalist feed "News" with URL "https://example.com/news.rss"
    And the feed has whitelist keywords "AI,technology"
    And the feed has blacklist keywords "crypto,cryptocurrency,blockchain"
    And the feed contains these articles:
      | title                        | should_match |
      | AI and Blockchain Tech       | no           |
      | New AI Model                 | yes          |
      | Crypto Markets Rally         | no           |
    When I execute Step 1
    Then Step 1 should succeed
    And 1 article should be accepted

  Scenario: Handle feed timeout gracefully
    Given a feed "SlowFeed" that times out
    And another feed "FastFeed" that responds successfully
    When I execute Step 1
    Then Step 1 should succeed
    And 1 feed should be fetched successfully
    And 1 feed should fail
    And the failure should be logged

  Scenario: Generate unique slugs for similar titles
    Given a specialized feed with these articles:
      | title                  |
      | Breaking AI News       |
      | Breaking AI News!      |
      | Breaking AI News Today |
    When I execute Step 1
    Then Step 1 should succeed
    And all 3 slugs should be unique
    And slug collisions should be handled

  Scenario: Handle malformed RSS feed
    Given a feed with invalid XML
    When I execute Step 1
    Then the feed should fail with parse error
    And the error should be logged
    And Step 1 should continue processing other feeds

  Scenario: Process multiple feeds in parallel
    Given 3 enabled feeds
    And parallel fetching is enabled
    When I execute Step 1
    Then all feeds should be fetched concurrently
    And Step 1 should complete successfully

  Scenario: Skip disabled feeds
    Given a feed "ActiveFeed" that is enabled
    And a feed "InactiveFeed" that is disabled
    When I execute Step 1
    Then only "ActiveFeed" should be fetched
    And "InactiveFeed" should be skipped

  Scenario: Cache results after successful ingestion
    Given a specialized feed with 10 articles
    When I execute Step 1
    Then Step 1 should succeed
    And the articles should be saved to cache
    And the cache should contain 10 articles

  Scenario: Handle feed with no articles
    Given a feed that returns empty content
    When I execute Step 1
    Then Step 1 should succeed
    And 0 articles should be returned
    And a warning should be logged

  # Scenario: Filter by RSS categories
  #   Given a generalist feed with category filtering
  #   And the feed has whitelist categories "AI"
  #   And the feed contains articles with these categories:
  #     | title        | categories    | should_match |
  #     | AI Article   | AI,Technology | yes          |
  #     | Sports News  | Sports        | no           |
  #   When I execute Step 1
  #   Then 1 article should be accepted
  # Skipped: Category filtering not yet implemented in Step 1
