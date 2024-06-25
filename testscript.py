class Test():
    classvar = 'asd'

    def __init__(self):
        self.classvar = 'xyz'

    def test(self):
        print(self.classvar)

Test().test()