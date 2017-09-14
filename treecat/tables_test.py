from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from treecat.tables import Table
from treecat.util import set_random_seed


@pytest.parametrize('num_rows_list', [
    [2],
    [1, 1],
    [2, 3],
    [0, 3, 1, 0, 2],
])
@pytest.parametrize('feature_types', [
    [],
    [TY_MULTINOMIAL],
    [TY_MULTINOMIAL, TY_MULTINOMIAL],
    [TY_CATEGORICAL],
    [TY_CATEGORICAL, TY_CATEGORICAL],
    [TY_MULTINOMIAL, TY_CATEGORICAL, TY_MULTINOMIAL, TY_CATEGORICAL],
    [TY_CATEGORICAL, TY_MULTINOMIAL, TY_CATEGORICAL, TY_MULTINOMIAL],
])
def test_table_concatenate(num_rows_list, feature_types):
    set_random_seed(0)
    num_rows = sum(num_rows_list)
    expected_table = generate_table(feature_types, num_rows)
    tables = []
    pos = 0
    for N in num_rows_list:
        tables.append(expected_table[pos:pos + N])
    actual_table = Table.concatenate(tables)
    assert actual_table == expected_table
