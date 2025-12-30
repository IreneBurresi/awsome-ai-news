Feature: Step 0 - Cache Management
  As a pipeline orchestrator
  I want to manage cache cleanup and backups
  So that the pipeline has fresh storage for new runs

  Background:
    Given the cache directory exists
    And the cache has retention policy of 10 days for articles and 3 days for news

  Scenario: Clean cache with no existing entries
    Given the cache is empty
    When Step 0 executes with cleanup enabled
    Then Step 0 should succeed
    And no cache entries should be cleaned
    And the cache directory should be ready for use

  Scenario: Clean expired cache entries
    Given the cache contains articles older than 10 days
    And the cache contains news older than 3 days
    And the cache contains fresh articles
    When Step 0 executes with cleanup enabled
    Then Step 0 should succeed
    And old articles should be removed
    And old news should be removed
    And fresh articles should remain in cache

  Scenario: Backup cache before cleanup
    Given the cache contains some entries
    When Step 0 executes with backup enabled
    Then Step 0 should succeed
    And a cache backup should be created
    And the backup should contain all original entries

  Scenario: Skip backup when disabled
    Given the cache contains some entries
    When Step 0 executes with backup disabled
    Then Step 0 should succeed
    And no cache backup should be created

  Scenario: Skip cleanup when disabled
    Given the cache contains old entries
    When Step 0 executes with cleanup disabled
    Then Step 0 should succeed
    And no cache entries should be cleaned
    And all old entries should remain

  Scenario: Handle cache directory creation
    Given the cache directory does not exist
    When Step 0 executes
    Then Step 0 should succeed
    And the cache directory should be created
    And the cache should be ready for use

  Scenario: Maintain only 5 most recent backups
    Given the cache has 6 existing backups
    When Step 0 executes with backup enabled
    Then Step 0 should succeed
    And exactly 5 backups should remain
    And the oldest backup should be removed
