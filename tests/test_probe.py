from deepwell import probe


def test_probe_schema_minimal():
    result = probe()
    assert isinstance(result, dict)
    assert "devices" in result and isinstance(result["devices"], list)
    assert "links" in result and isinstance(result["links"], list)
    assert "hbw_gbps" in result and isinstance(result["hbw_gbps"], dict)


