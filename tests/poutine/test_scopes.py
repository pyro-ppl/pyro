import pyro
import pyro.poutine as poutine
import pyro.distributions.torch as dist

from pyro.poutine import scope


def test_recur_multi():

    @scope
    def model1(r=True):
        model2()
        with scope(prefix="inter"):
            model2()
            if r:
                model1(r=False)
        model2()

    @scope
    def model2():
        return pyro.sample("y", dist.Normal(0.0, 1.0))

    tr = poutine.trace(model1).get_trace()
    print(tr)
    for name in tr:
        print(name)


def test_only_withs():

    def model1():
        with scope(prefix="a"):
            with scope(prefix="b"):
                pyro.sample("x", dist.Bernoulli(0.5))

    tr1 = poutine.trace(model1).get_trace()
    assert "a/b/x" in tr1.nodes

    tr2 = poutine.trace(scope(prefix="model1")(model1)).get_trace()
    assert "model1/a/b/x" in tr2.nodes


def test_mutual_recur():

    @scope
    def model1(n):
        pyro.sample("a", dist.Bernoulli(0.5))
        if n <= 0:
            return
        else:
            return model2(n-1)

    @scope
    def model2(n):
        pyro.sample("b", dist.Bernoulli(0.5))
        if n <= 0:
            return
        else:
            model1(n)

    names = set(["_INPUT", "_RETURN",
                 "model2/b", "model2/model1/a", "model2/model1/model2/b"])
    tr_names = set([name for name in poutine.trace(model2).get_trace(1)])
    assert names == tr_names


def test_simple_recur():

    @scope
    def geometric(p):
        x = pyro.sample("x", dist.Bernoulli(p))
        if x.item() == 1.0:
            # model1()
            return x + geometric(p)
        else:
            return x

    prev_name = "x"
    for name, node in poutine.trace(geometric).get_trace(0.9).nodes.items():
        if node["type"] == "sample":
            print(name)
            assert name == "geometric/" + prev_name
            prev_name = "geometric/" + prev_name


def test_basic_scope():

    @scope
    def f1():
        return pyro.sample("x", dist.Bernoulli(0.5))

    @scope
    def f2():
        f1()
        return pyro.sample("y", dist.Bernoulli(0.5))

    tr1 = poutine.trace(f1).get_trace()
    assert "f1/x" in tr1.nodes

    tr2 = poutine.trace(f2).get_trace()
    assert "f2/f1/x" in tr2.nodes
    assert "f2/y" in tr2.nodes
