import re

def validate_name(full_name):
    # Logic from main_fastapi.py
    if not re.search(r'^[A-Za-z\s]{2,}$', full_name) or not any(c.isalpha() for c in full_name):
        return False, "Full name must contain at least 2 letters and no special characters or numbers"
    return True, "Valid"

test_cases = [
    ("Alice Smith", True),
    ("Bo", True),
    ("123", False),
    ("A", False),
    ("Alice123", False),
    ("John Doe!", False),
    ("  ", False),
    ("Jo ", True),
]

for name, expected in test_cases:
    is_valid, msg = validate_name(name)
    status = "PASS" if is_valid == expected else "FAIL"
    print(f"Testing '{name}': {status} (Expected: {expected}, Got: {is_valid}, Msg: {msg})")
