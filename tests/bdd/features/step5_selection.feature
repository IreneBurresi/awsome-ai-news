Feature: Step 5 - Top News Selection and Categorization
  As a news pipeline
  I want to categorize news and select the top 10 most important items
  So that users see the most relevant AI news

  Background:
    Given Step 5 is enabled
    And the target count is 10
    And the minimum quality score is 0.6

  Scenario: Categorize and score all news
    Given we have 5 news clusters
    And the Gemini API categorizes them successfully
    When I run Step 5
    Then Step 5 should succeed
    And all 5 news should be categorized
    And the top news should be sorted by importance score
    And the API should be called 1 time

  Scenario: Select top 10 from larger set
    Given we have 20 news clusters
    And the Gemini API categorizes them with scores from 10.0 to 4.0
    When I run Step 5
    Then Step 5 should succeed
    And all 20 news should be categorized
    And only the top 10 news should be selected
    And the top news scores should be descending

  Scenario: Handle partial API categorization
    Given we have 5 news clusters
    And the Gemini API only categorizes 3 of them
    When I run Step 5
    Then Step 5 should succeed
    And all 5 news should be categorized
    And 2 news should have default category "other"
    And 2 news should have default score 5.0

  Scenario: Calculate category distribution
    Given we have 6 news clusters
    And 2 are categorized as "model_release"
    And 2 are categorized as "research"
    And 1 is categorized as "policy_regulation"
    And 1 is categorized as "industry_news"
    When I run Step 5
    Then the category distribution should show 2 model_release
    And the category distribution should show 2 research
    And the category distribution should show 1 policy_regulation

  Scenario: Clamp out-of-range scores
    Given we have 3 news clusters
    And one has importance score 15.0
    And one has importance score -5.0
    And one has importance score 7.0
    When I run Step 5
    Then the first score should be clamped to 10.0
    And the second score should be clamped to 0.0
    And the third score should remain 7.0

  Scenario: No news to categorize
    Given we have 0 news clusters
    When I run Step 5
    Then Step 5 should succeed
    And no news should be selected
    And the API should not be called
