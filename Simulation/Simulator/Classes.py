class Class:
    def __init__(self, shape_size: bool, movement: bool, touch_type: bool, dangerous: bool):
        self.big: bool = shape_size
        self.moving: bool = movement
        self.press: bool = touch_type
        self.dangerous: bool = dangerous
