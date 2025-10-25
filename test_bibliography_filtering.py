#!/usr/bin/env python3
"""
Test bibliography detection and filtering.

This script tests the new bibliography filtering functionality to ensure
oversized bibliography chunks are properly excluded.
"""

from src.fileintel.document_processing.element_detection import is_bibliography_section

# Test cases from actual oversized chunks
test_cases = {
    "Bibliography - Example 1": {
        "text": """Coupland, D. (2014), "The Ghost of Invention: A Visit to Bell Labs", available at http://www.wired. com/2014/09/coupland-bell-labs/ (accessed 30 October 2015). Damanpour, F. (1991), "Organizational Innovation: A Meta-analysis of Effects of Determinants and Moderators", Academy of Management Journal, 34 (3), 555–90. Day, G.S. (2007), "Is it Real? Can We Win? Is it Worth Doing?", Harvard Business Review, 85 (12), 110–20. Dotzel, T., V. Shankar and L.L. Berry (2013), "Service Innovativeness and Firm Value", Journal of Marketing, 77 (1), 76–97.""",
        "section_title": None,
        "section_path": None,
        "expected": True
    },

    "Bibliography - Example 2": {
        "text": """Ryan, R. M., La Guardia, J. G., Solky-Butzel, J., Chirkov, V., & Kim, Y. (2005). On the interpersonal regulation of emotions: Emotional reliance across gender, relationships, and culture. Personal Relationships, 12, 146163. Ryan, R. M., & Lynch, J. (1989). Emotional autonomy versus detachment: Revisiting the vicissitudes of adolescence and young adulthood. Child Development, 60, 340356. Ryan, R. M., Mims, V., & Koestner, R. (1983). Relation of reward contingency and interpersonal context to intrinsic motivation: A review and test using cognitive evaluation theory.""",
        "section_title": None,
        "section_path": None,
        "expected": True
    },

    "Bibliography - With Section Title": {
        "text": """Smith, J. (2020). Paper title. Journal Name, 10(2), 123-145.""",
        "section_title": "References",
        "section_path": "8. References",
        "expected": True
    },

    "Bibliography - Works Cited": {
        "text": """Jones, K. (2019). Another paper. Conference Proceedings.""",
        "section_title": "Works Cited",
        "section_path": None,
        "expected": True
    },

    "Regular Text - Agile": {
        "text": """Agile is a software development methodology that emphasizes iterative development, collaboration, and flexibility. The approach was formalized in the Agile Manifesto, which outlines core principles for software development. Teams using Agile work in short sprints, typically 1-4 weeks long.""",
        "section_title": "Introduction",
        "section_path": "1. Introduction",
        "expected": False
    },

    "Regular Text - With Citations": {
        "text": """As Smith (2020) noted, agile methodologies have become increasingly popular. The study by Jones et al. (2019) further demonstrated this trend. However, as noted in the literature review, there are challenges.""",
        "section_title": "Literature Review",
        "section_path": "2. Literature Review",
        "expected": False
    },

    "Short Text - Too Short": {
        "text": """Brief text.""",
        "section_title": None,
        "section_path": None,
        "expected": False
    }
}


def test_bibliography_detection():
    """Test bibliography detection on all test cases."""

    print("=" * 80)
    print("Testing Bibliography Detection")
    print("=" * 80)

    passed = 0
    failed = 0

    for name, case in test_cases.items():
        result = is_bibliography_section(
            case["text"],
            case["section_title"],
            case["section_path"]
        )

        expected = case["expected"]
        status = "✓ PASS" if result == expected else "✗ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} | {name}")
        print(f"  Expected: {expected}, Got: {result}")
        if case["section_title"]:
            print(f"  Section: {case['section_title']}")
        print(f"  Text length: {len(case['text'])} chars")

        if result != expected:
            print(f"  Text preview: {case['text'][:100]}...")

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = test_bibliography_detection()
    exit(0 if success else 1)
