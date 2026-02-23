from garden_ml.features.schema import SCHEMA


def test_schema_total_is_188():
    assert SCHEMA.total == 188


def test_schema_parts_sum_to_total():
    assert sum(SCHEMA.parts.values()) == SCHEMA.total


def test_schema_order_matches_parts():
    assert set(SCHEMA.order) == set(SCHEMA.parts.keys())
