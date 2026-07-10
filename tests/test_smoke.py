import vet_agent


def test_package_imports_and_has_version():
    assert vet_agent.__version__ == "0.1.0"
