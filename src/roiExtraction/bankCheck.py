# ---Libraries---
# Standard library

# Third-party libraries
import cv2
from scipy.misc import imresize

# Private libraries
import algorithms.slant_angle as slant_angle
import check_input_fields as check_input_fields


class BankCheck:

    def __init__(self, path):
        self.checkImage = slant_angle.fix_check(path)
        self.checkImage = imresize(self.checkImage, (550, 1240))

    def input_fields(self):
        return check_input_fields.extract(self.checkImage)

    def clean_input_fields(self):
        return check_input_fields.clean(self.checkImage)

    def amount_field(self, clean=True):
        return check_input_fields.extractAmount(self.inputFields(), clean)

    def date_field(self):
        return check_input_fields.extractDate(self.inputFields())

    def save_input_fields(self, name, clean=False):
        if clean:
            cv2.imwrite(name, self.cleanInputFields())
        else:
            cv2.imwrite(name, self.inputFields())

    def save_check(self, name):
        cv2.imwrite(name, self.checkImage)