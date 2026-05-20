from auth import check_and_log

print("=" * 45)
print("  LPR Gate System — Auth Function Test")
print("=" * 45)

# Test 1: plate that EXISTS in your vehicles table
result1 = check_and_log(
    plate_number="KL07BH1234",
    confidence_score=92.5
)
print("\n[TEST 1] Known plate:")
print(result1)

# Test 2: plate that DOES NOT exist
result2 = check_and_log(
    plate_number="KL05AX9999",
    confidence_score=87.3
)
print("\n[TEST 2] Unknown plate:")
print(result2)

# Test 3: EXIT of known plate
result3 = check_and_log(
    plate_number="KL09CA5678",
    confidence_score=95.0
)
print("\n[TEST 3] Known plate EXIT:")
print(result3)

print("\n" + "=" * 45)
print("Check pgAdmin → gate_logs to verify rows")
print("=" * 45)