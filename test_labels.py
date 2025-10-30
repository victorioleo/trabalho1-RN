"""
Test script to verify CIFAR-100 label mappings.

This verifies that:
1. All 100 fine labels map to exactly one coarse label
2. Each coarse label has exactly 5 fine labels
3. The mapping is consistent
"""
from collections import Counter
from cifar100_labels import (
    CIFAR100_FINE_TO_COARSE,
    COARSE_LABELS,
    FINE_LABELS,
    get_coarse_label
)


def test_label_mapping():
    print("Testing CIFAR-100 label mappings...")
    print("=" * 60)
    
    # Test 1: All fine labels have a mapping
    print("\n1. Checking all 100 fine labels are mapped...")
    assert len(CIFAR100_FINE_TO_COARSE) == 100, "Should have 100 fine labels"
    for i in range(100):
        assert i in CIFAR100_FINE_TO_COARSE, f"Fine label {i} missing from mapping"
    print("   ✓ All 100 fine labels are mapped")
    
    # Test 2: All coarse labels are in range [0, 19]
    print("\n2. Checking coarse labels are in valid range...")
    coarse_values = set(CIFAR100_FINE_TO_COARSE.values())
    assert coarse_values == set(range(20)), "Should have exactly coarse labels 0-19"
    print("   ✓ Coarse labels are in range [0, 19]")
    
    # Test 3: Each coarse label has exactly 5 fine labels
    print("\n3. Checking each coarse label has 5 fine labels...")
    coarse_counts = Counter(CIFAR100_FINE_TO_COARSE.values())
    for coarse_id in range(20):
        count = coarse_counts[coarse_id]
        assert count == 5, f"Coarse label {coarse_id} has {count} fine labels, expected 5"
    print("   ✓ Each coarse label has exactly 5 fine labels")
    
    # Test 4: get_coarse_label function works correctly
    print("\n4. Testing get_coarse_label function...")
    for fine_id, expected_coarse_id in CIFAR100_FINE_TO_COARSE.items():
        actual_coarse_id = get_coarse_label(fine_id)
        assert actual_coarse_id == expected_coarse_id, \
            f"get_coarse_label({fine_id}) = {actual_coarse_id}, expected {expected_coarse_id}"
    print("   ✓ get_coarse_label function works correctly")
    
    # Test 5: Label lists have correct lengths
    print("\n5. Checking label list lengths...")
    assert len(COARSE_LABELS) == 20, f"Should have 20 coarse labels, got {len(COARSE_LABELS)}"
    assert len(FINE_LABELS) == 100, f"Should have 100 fine labels, got {len(FINE_LABELS)}"
    print("   ✓ Label lists have correct lengths")
    
    # Print example mappings
    print("\n" + "=" * 60)
    print("Example Mappings (first 3 superclasses):")
    print("=" * 60)
    for coarse_id in range(3):
        fine_ids = [fine_id for fine_id, cid in CIFAR100_FINE_TO_COARSE.items() 
                   if cid == coarse_id]
        fine_ids.sort()
        print(f"\nSuperclass {coarse_id}: {COARSE_LABELS[coarse_id]}")
        for fine_id in fine_ids:
            print(f"  - Fine class {fine_id:2d}: {FINE_LABELS[fine_id]}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_label_mapping()
