"""
Test export/import compatibility for metadata.

Verifies that metadata can round-trip through CSV export/import correctly.
"""

def test_compatibility():
    """Test that supported metadata types round-trip correctly."""

    # Original metadata with various types
    original = {
        "title": "Research Paper",                      # String
        "author": "Smith, John",                        # String with comma
        "authors": ["Smith, John", "Jones, Mary"],      # List of strings
        "year": "2023",                                 # String (number)
        "year_int": 2023,                               # Actual int
        "keywords": ["AI", "ML", "Data Science"],       # List
        "abstract": "This is a paper about AI, ML, and more.",  # String with commas
        "nested": {"type": "journal", "volume": 5},     # Dict (should be excluded)
    }

    print("=" * 80)
    print("ORIGINAL METADATA:")
    print("=" * 80)
    for key, value in original.items():
        print(f"{key:15} ({type(value).__name__:10}): {value}")

    # Simulate export (what gets written to CSV)
    from src.fileintel.cli.formatters import _serialize_value_for_csv

    exported = {}
    for key, value in original.items():
        if not isinstance(value, dict):  # Dicts excluded
            exported[key] = _serialize_value_for_csv(value)

    print("\n" + "=" * 80)
    print("EXPORTED TO CSV (all become strings):")
    print("=" * 80)
    for key, value in exported.items():
        print(f"{key:15}: {repr(value)}")

    # Simulate import (what gets parsed from CSV)
    from src.fileintel.cli.formatters import _parse_csv_value

    imported = {}
    for key, value in exported.items():
        imported[key] = _parse_csv_value(value)

    print("\n" + "=" * 80)
    print("IMPORTED FROM CSV:")
    print("=" * 80)
    for key, value in imported.items():
        print(f"{key:15} ({type(value).__name__:10}): {value}")

    # Verify compatibility
    print("\n" + "=" * 80)
    print("COMPATIBILITY CHECK:")
    print("=" * 80)

    checks = {
        "title": original["title"] == imported["title"],
        "author": original["author"] == imported["author"],
        "authors": original["authors"] == imported["authors"],
        "year": original["year"] == imported["year"],  # String year
        "year_int": str(original["year_int"]) == imported["year_int"],  # Int→String
        "keywords": original["keywords"] == imported["keywords"],
        "abstract": original["abstract"] == imported["abstract"],
        "nested": "nested" not in imported,  # Dict excluded
    }

    for field, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{field:15}: {status}")

    all_passed = all(checks.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Export/import is compatible!")
    else:
        print("✗ SOME CHECKS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/tuomo/code/fileintel")

    success = test_compatibility()
    sys.exit(0 if success else 1)
