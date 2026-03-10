SARABCRAFT_R1_NAME = "SarabCraft R1"
def canonicalize_attack_name(name):
    return name


def _normalize_attack_list(value):
    if isinstance(value, list):
        return [canonicalize_attack_name(item) for item in value]
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return json.dumps([canonicalize_attack_name(item) for item in parsed])
    return value


def normalize_attack_payload(value):
    if isinstance(value, list):
        return [normalize_attack_payload(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if key in {"attack", "best_attack", "attack_name"}:
                normalized[key] = canonicalize_attack_name(child)
            elif key in {"attacks", "selected_attacks"}:
                normalized[key] = _normalize_attack_list(child)
            elif key == "attacks_json":
                normalized[key] = _normalize_attack_list(child)
            else:
                normalized[key] = normalize_attack_payload(child)
        return normalized
    return value
