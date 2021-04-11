class Shape:
    def __init__(self):
        pass


class Ellipse(Shape):
    # ellipse formula: (x-h)**2/a**2 + (y-k)**2/b**2 = 1
    # h, k -> center
    # a, b -> width, height

    def __init__(self, h, k, a, b, force):
        super().__init__()
        self.h = h
        self.k = k
        self.a = a
        self.b = b
        self.force = force

    def is_in(self, x, y):
        val = (x - self.h) ** 2 / self.a ** 2 + (y - self.k) ** 2 / self.b ** 2
        if val <= 1:
            return True
        else:
            return False
