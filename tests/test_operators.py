import pytest
from hypothesis import given
from hypothesis.strategies import lists
from minitorch import MathTest
from minitorch.operators import (
    add, addLists, eq, id, inv, inv_back, log_back, lt, max, mul,
    neg, negList, prod, relu, relu_back, sigmoid, sum,
)
from .strategies import assert_close, small_floats


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a, b):
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a):
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a):
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a):
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a):
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    assert 0.0 <= sigmoid(a) <= 1.0
    assert_close(1.0 - sigmoid(a), sigmoid(-a))
    assert sigmoid(0.0) == 0.5
    assert sigmoid(a) <= sigmoid(a + 1.0)
    if -20 <= a <= 20:
        assert sigmoid(a) < sigmoid(a + 1.0)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    if lt(a, b) and lt(b, c):
        assert lt(a, c)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a, b):
    assert mul(a, b) == mul(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))


@pytest.mark.task0_2
@given(small_floats)
def test_other(a):
    assert neg(neg(a)) == a
    assert add(a, neg(a)) == 0.0


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    x1, x2 = addLists([a, b], [c, d])
    assert_close(x1, a + c)
    assert_close(x2, b + d)


@pytest.mark.task0_3
@given(lists(small_floats, min_size=5, max_size=5),
       lists(small_floats, min_size=5, max_size=5))
def test_sum_distribute(ls1, ls2):
    assert_close(sum(ls1) + sum(ls2), sum(addLists(ls1, ls2)))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls):
    for i, j in zip(ls, negList(ls)):
        assert_close(i, -j)


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, t1, t2):
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a, b):
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
