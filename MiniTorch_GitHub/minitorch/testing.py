from . import operators


class MathTest:
    @staticmethod
    def neg(a):
        return -a
    @staticmethod
    def addConstant(a):
        return 5 + a
    @staticmethod
    def square(a):
        return a * a
    @staticmethod
    def cube(a):
        return a * a * a
    @staticmethod
    def subConstant(a):
        return a - 5
    @staticmethod
    def multConstant(a):
        return 5 * a
    @staticmethod
    def div(a):
        return a / 5
    @staticmethod
    def inv(a):
        return operators.inv(a + 3.5)
    @staticmethod
    def sig(a):
        return operators.sigmoid(a)
    @staticmethod
    def log(a):
        return operators.log(a + 100000)
    @staticmethod
    def relu(a):
        return operators.relu(a + 5.5)
    @staticmethod
    def exp(a):
        return operators.exp(a - 200)
    @staticmethod
    def explog(a):
        return operators.log(a + 100000) + operators.exp(a - 200)
    @staticmethod
    def add2(a, b):
        return a + b
    @staticmethod
    def mul2(a, b):
        return a * b
    @staticmethod
    def div2(a, b):
        return a / (b + 5.5)
    @staticmethod
    def gt2(a, b):
        return operators.lt(b, a + 1.2)
    @staticmethod
    def lt2(a, b):
        return operators.lt(a + 1.2, b)
    @staticmethod
    def eq2(a, b):
        return operators.eq(a, b + 5.5)
    @staticmethod
    def sum_red(a):
        return operators.sum(a)
    @staticmethod
    def mean_red(a):
        return operators.sum(a) / float(len(a))
    @staticmethod
    def mean_full_red(a):
        return operators.sum(a) / float(len(a))
    @staticmethod
    def complex(a):
        return operators.log(operators.sigmoid(operators.relu(
            operators.relu(a * 10 + 7) * 6 + 5) * 10)) / 50
    @classmethod
    def _tests(cls):
        one, two, red = [], [], []
        for name in dir(MathTest):
            if name.startswith("_") or not callable(getattr(MathTest, name)):
                continue
            target = two if name.endswith("2") else red if name.endswith("red") else one
            target.append((name, getattr(cls, name)))
        return one, two, red
    @classmethod
    def _comp_testing(cls):
        return tuple([(name, base_fn, fn)
                      for (name, fn), (_, base_fn) in zip(own, base)]
                     for own, base in zip(cls._tests(), MathTest._tests()))


class MathTestVariable(MathTest):
    @staticmethod
    def inv(a):
        return 1.0 / (a + 3.5)
    @staticmethod
    def sig(x):
        return x.sigmoid()
    @staticmethod
    def log(x):
        return (x + 100000).log()
    @staticmethod
    def relu(x):
        return (x + 5.5).relu()
    @staticmethod
    def exp(a):
        return (a - 200).exp()
    @staticmethod
    def explog(a):
        return (a + 100000).log() + (a - 200).exp()
    @staticmethod
    def sum_red(a):
        return a.sum(0)
    @staticmethod
    def mean_red(a):
        return a.mean(0)
    @staticmethod
    def mean_full_red(a):
        return a.mean()
    @staticmethod
    def eq2(a, b):
        return a == b + 5.5
    @staticmethod
    def gt2(a, b):
        return a + 1.2 > b
    @staticmethod
    def lt2(a, b):
        return a + 1.2 < b
    @staticmethod
    def complex(a):
        return (((a * 10 + 7).relu() * 6 + 5).relu() * 10).sigmoid().log() / 50
