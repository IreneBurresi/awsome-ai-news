Feature: Step 4 - Multi-day News Deduplication
  As a news pipeline
  I want to deduplicate news across multiple days
  So that users don't see the same news repeatedly

  Background:
    Given Step 4 is enabled
    And the lookback window is 3 days
    And the similarity threshold is 0.85

  Scenario: First run with no cached news
    Given today we have 2 news clusters
    And there is no cached news from previous days
    When I run Step 4
    Then Step 4 should succeed
    And the result should contain 2 unique news
    And 0 duplicates should be found
    And the news should be saved to cache

  Scenario: Second run finds duplicate news
    Given today we have 1 news cluster about "GPT-5 Release"
    And yesterday we had 1 news cluster about "OpenAI GPT-5 Launch"
    And the Gemini API identifies them as duplicates
    When I run Step 4
    Then Step 4 should succeed
    And 1 duplicate should be found
    And 1 news should be merged
    And the merged news should have 2 article slugs
    And the API should be called 1 time

  Scenario: No duplicates found across days
    Given today we have 2 news clusters
    And yesterday we had 3 news clusters
    And the Gemini API finds no duplicates
    When I run Step 4
    Then Step 4 should succeed
    And 0 duplicates should be found
    And the result should contain 5 unique news
    And all original news should be preserved

  Scenario: API failure with fallback enabled
    Given today we have 2 news clusters
    And yesterday we had 1 news cluster
    And the Gemini API will fail
    And fallback is enabled
    When I run Step 4
    Then Step 4 should succeed with fallback
    And the result should contain 2 unique news
    And the fallback flag should be true
    And all today's news should be preserved

  Scenario: Lookback window filters old cache
    Given today we have 1 news cluster
    And we have cached news from 2 days ago
    And we have cached news from 5 days ago
    When I run Step 4
    Then only news from 2 days ago should be loaded
    And news from 5 days ago should be filtered out

  Scenario: Merge keeps news with more articles
    Given today we have 1 news cluster with 3 articles
    And yesterday we had 1 news cluster with 1 article about the same topic
    And the Gemini API identifies them as duplicates
    When I run Step 4
    Then the merged news should use today's news as base
    And the merged news should have 4 articles total
