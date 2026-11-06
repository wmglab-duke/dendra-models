"""Public morphology catalogs describe the resources shipped with each model."""

from importlib.resources import files

import dendra as dn  # noqa: F401 - configure Dendra before importing the models
import pytest

from dendra_models.models.cells.cortical import L23_PC_cADpyr, L5_TTPC2_cADpyr


CATALOGS = (
    (L23_PC_cADpyr, "L23", "L23_PC_cADpyr", 44),
    (L5_TTPC2_cADpyr, "L5", "L5_TTPC2_cADpyr", 37),
)


@pytest.mark.parametrize("constructor,layer,name,count", CATALOGS, ids=["L23", "L5"])
def test_valid_ids_cover_every_packaged_morphology(constructor, layer, name, count):
    ids = constructor.valid_ids()
    assert len(ids) == count
    assert ids == sorted(set(ids))
    assert all(type(identifier) is int for identifier in ids)
    assert 1 in ids

    directory = files(f"dendra_models.models.cells.cortical.{layer}") / name
    packaged = {path.name for path in directory.iterdir() if path.name.endswith(".gml")}
    assert packaged == {f"{name}_{identifier}.gml" for identifier in ids}
    assert all((directory / filename).is_file() for filename in packaged)


@pytest.mark.parametrize("constructor,layer,name,count", CATALOGS, ids=["L23", "L5"])
def test_missing_morphology_reports_the_available_ids(constructor, layer, name, count):
    with pytest.raises(FileNotFoundError) as error:
        constructor(-1)
    assert f"{name}_-1.gml" in str(error.value)
    assert f"Valid IDs: {constructor.valid_ids()}" in str(error.value)
