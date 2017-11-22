# ---Libraries---
# Standard library
import os
import sys
import math

# Third-party libraries
import cv2
import numpy as np
import scipy.ndimage as ndimage

# Private libraries
sys.path.append(os.path.abspath("../"))
import utils


template_png='algorithms/inputFields/template.png'
amount_input_png='algorithms/inputFields/amount_template.png'
date_input_png='algorithms/inputFields/date_template.png'


def searchTemplateCenterPointIn(check, template, searchMap, step=1, threshold=-9999999):
    fromIndex = [int(template.shape[0] / 2 + 1), int(template.shape[1] / 2 + 1)]
    toIndex = [int(searchMap.shape[0] - template.shape[0] / 2), int(searchMap.shape[1] - template.shape[1] / 2)]

    radios = [int(template.shape[0] / 2), int(template.shape[1] / 2)]

    maxConv = threshold
    maxCenterConv = [0, 0]

    for centerConvX in range(fromIndex[0], toIndex[0]):
        for centerConvY in range(fromIndex[1], toIndex[1]):
            if searchMap[centerConvX, centerConvY] == 1:
                convMatrix = check[centerConvX - radios[0]:centerConvX + radios[0] + template.shape[0]%2,
                             centerConvY - radios[1]:centerConvY + radios[1] + template.shape[1]%2] \
                             * template
                conv = np.sum(convMatrix)
                if maxConv < conv:
                    maxConv = conv
                    maxCenterConv = [centerConvX, centerConvY]

    print maxConv
    return maxCenterConv


def normalize(image):
    binary = np.array(image, dtype=np.int8, copy=True)

    binary[image == 0] = 1
    binary[image == 255] = -1

    return binary


def binaryTemplate():
    img_template = cv2.imread(template_png)
    return utils.sanitize(img_template)


def dateTemplate():
    img_template = cv2.imread(date_input_png)
    return utils.sanitize(img_template)


def amountTemplate():
    img_template = cv2.imread(amount_input_png)
    return utils.sanitize(img_template)


def binaryTemplateFix():
    img_template = cv2.imread(template_png)
    return utils.sanitize(img_template, False)


# Extract input fields, the Region Of Interest (ROI), from bank check.
def extract(check):
    template = binaryTemplate()
    templateRadios = [template.shape[0] / 2, template.shape[1] / 2]

    checkMap = np.array(check, dtype=np.int8)

    checkMap[check == 0] = 1
    checkMap[check > 0] = -1

    searchFrom = [check.shape[0] / 2 - 10, check.shape[1] / 2 - 10]
    searchTo = [check.shape[0] / 2 + 100, check.shape[1] / 2 + 10]

    searchMatrix = np.zeros(check.shape, np.uint8)
    searchMatrix[int(searchFrom[0]):int(searchTo[0]), int(searchFrom[1]):int(searchTo[1])] = 1

    center = searchTemplateCenterPointIn(checkMap, template, searchMatrix)

    inputFieldsRectangle = [[int(center[0] - templateRadios[0] - 1), int(center[0] + templateRadios[0])],
                            [int(center[1] - templateRadios[1]), int(center[1] + templateRadios[1])]]

    roi = check[inputFieldsRectangle[0][0]:inputFieldsRectangle[0][1],
                inputFieldsRectangle[1][0]:inputFieldsRectangle[1][1]]

    return roi


def extractAmount(input_fields, clean = True):
    template = amountTemplate()

    template[template == -1] = 0

    input_fields_map = normalize(input_fields)

    amountX = 1018
    amountY = 96

    searchFrom = [amountY - 60, amountX - 60]
    searchTo = [amountY + 60, amountX + 60]

    searchMatrix = np.zeros(input_fields.shape, np.uint8)
    searchMatrix[int(searchFrom[0]):int(searchTo[0]), int(searchFrom[1]):int(searchTo[1])] = 1


    center = searchTemplateCenterPointIn(input_fields_map, template, searchMatrix)

    inputFieldsRectangle = [[int(center[0] - template.shape[0]/2), int(center[0] + template.shape[0]/2)],
                            [int(center[1] - template.shape[1]/2), int(center[1] + template.shape[1]/2)]]

    template[template == 0] = -1
    template[template == 1] = 0
    template[:,0:35] = 0
    input_fields_clean = cleanBy(input_fields[inputFieldsRectangle[0][0]:inputFieldsRectangle[0][1],
          inputFieldsRectangle[1][0]:inputFieldsRectangle[1][1]], template)

    inputFieldsRectangle[1][1] = input_fields.shape[1] if inputFieldsRectangle[1][1] + 50 > input_fields.shape[1] \
        else inputFieldsRectangle[1][1] + 50

    inputFieldsRectangle[0][0] -= 20

    roi = np.copy(input_fields[inputFieldsRectangle[0][0]:inputFieldsRectangle[0][1],
          inputFieldsRectangle[1][0]:inputFieldsRectangle[1][1]])

    if clean:
        roi[20:roi.shape[0], 0:input_fields_clean.shape[1]] = input_fields_clean

    return roi

def extractDate(input_fields):
    template = dateTemplate()

    input_fields_map = normalize(input_fields)

    amountX = 683
    amountY = 190

    searchFrom = [amountY - 100, amountX - 100]
    searchTo = [amountY + 100, amountX + 100]

    searchMatrix = np.zeros(input_fields.shape, np.uint8)
    searchMatrix[int(searchFrom[0]):int(searchTo[0]), int(searchFrom[1]):int(searchTo[1])] = 1

    center = searchTemplateCenterPointIn(input_fields, template, searchMatrix)

    inputFieldsRectangle = [[int(center[0] - 50), int(center[0] + 50)],
                            [int(center[1] - 113), int(center[1] + 113)]]

    roi = input_fields[inputFieldsRectangle[0][0]:inputFieldsRectangle[0][1],
          inputFieldsRectangle[1][0]:inputFieldsRectangle[1][1]]

    return roi


def clean(check):
    input_fields = extract(check)
    input_fields_OBIFs = compute_OBIFs.computeOBIFs(input_fields)
    empty_input_fields = binaryTemplateFix()
    empty_input_fields_OBIFs = compute_OBIFs.computeOBIFs(empty_input_fields)

    # input_fields[diff_map_not] = 255
    input_fields_clone = cleanBy(input_fields, empty_input_fields)
    # clean_input_fields_OBIFs = compute_OBIFs.computeOBIFs(input_fields)


    diff_map = np.equal(input_fields_OBIFs, empty_input_fields_OBIFs)
    # diff_map_clean = np.equal(input_fields_OBIFs, clean_input_fields_OBIFs)
    # diff_map_not = np.not_equal(input_fields_OBIFs, empty_input_fields_OBIFs)

    # input_fields_OBIFs[diff_map] = 30
    # empty_input_fields_OBIFs[diff_map] = 30

    if_obifs_color = color_BIFs.bifs_to_color_image(input_fields_OBIFs)
    eif_obifs_color = color_BIFs.bifs_to_color_image(empty_input_fields_OBIFs)
    # cif_obifs_color = color_BIFs.bifs_to_color_image(clean_input_fields_OBIFs)

    if_obifs_color[diff_map] = 30
    if_obifs_color[empty_input_fields_OBIFs == 0] = 30
    eif_obifs_color[diff_map] = 30
    # cif_obifs_color[diff_map_clean] = 30

    cv2.imwrite("obifInput.png", if_obifs_color)
    cv2.imwrite("obifEmptyInput.png", eif_obifs_color)
    # cv2.imwrite("obifCleanInput.png", cif_obifs_color)

    # diff_map[empty_input_fields != 0] = False

    return input_fields_clone


def cleanBy(image, template_image):

    image_clone = np.copy(image)
    image_clone[template_image == 0] = 255

    # kernel = np.zeros((5, 5), np.float16)
    # kernel[1][1] = 1/6.
    # kernel[1][2] = 1/6.
    # kernel[1][3] = 1/6.
    # kernel[3][2] = 1/6.
    # kernel[3][2] = 1/6.
    # kernel[3][3] = 1/6.
    #
    #
    # pixel_matrix = ndimage.filters.convolve(image_clone, kernel, mode='constant')
    # cv2.imwrite('test1.png', pixel_matrix)
    #
    # pixel_matrix[template_image != 0] = 255


    return image_clone

# Test
# img_template = cv2.imread('inputFields/templateFix1.png')
#
# image = np.array(img_template, dtype=np.uint8)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret3, invers1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, invers2 = cv2.threshold(invers1, 127, 255, cv2.THRESH_BINARY_INV)
# blur1 = cv2.GaussianBlur(image, (11, 11), 0)
# blur2 = cv2.GaussianBlur(image, (21, 21), 0)
# blur3 = cv2.GaussianBlur(image, (31, 31), 0)
# blur4 = cv2.GaussianBlur(image, (41, 41), 0)
#
#
# blur1 = np.array(blur1, dtype=np.uint8)
# blur2 = np.array(blur2, dtype=np.uint8)
# blur3 = np.array(blur3, dtype=np.uint8)
# blur4 = np.array(blur4, dtype=np.uint8)
#
# blur1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
# blur2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)
# blur3 = cv2.cvtColor(blur3, cv2.COLOR_BGR2GRAY)
# blur4 = cv2.cvtColor(blur4, cv2.COLOR_BGR2GRAY)
#
# ret3, invers1 = cv2.threshold(blur1, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret3, invers2 = cv2.threshold(blur2, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret3, invers3 = cv2.threshold(blur3, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret3, invers4 = cv2.threshold(blur4, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# cv2.imwrite('inputFields/templateV1.png', invers1)
# cv2.imwrite('inputFields/templateV2.png', invers2)
# cv2.imwrite('inputFields/templateV3.png', invers3)
# cv2.imwrite('inputFields/templateV4.png', invers4)
# cv2.imwrite('inputFields/template.png', invers1)


# img_template = cv2.imread('inputFields/templateFix1.png')
#
# image = np.array(img_template, dtype=np.uint8)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret3, invers1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#
# cv2.imwrite('inputFields/templateFix1.png', invers1)

# Create the template function ROI Weight function
# The max value indicate on the ROI position.
#
# def checkTemplate(name):
#     img_template = cv2.imread('inputFields/template.png')
#     image = np.array(img_template, dtype=np.uint8)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret3, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     binary = np.array(binary, dtype=np.int8)
#
#     binary[binary == 0] = 1
#     binary[binary == 255] = -1
#
#     img_template = cv2.imread('../../assets/Checks/' + name.__str__() + '.png')
#     image = utils.rotate_image_by_angle(img_template, -2.213)
#     image = np.array(image, dtype=np.uint8)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret3, check = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     checkMap = np.array(check, dtype=np.int8)
#
#     checkMap[check == 0] = 1
#     checkMap[check == 255] = -1
#
#     selectionFrom = [check.shape[0] / 2 - 10, check.shape[1] / 2 - 10]
#     selectionTo = [check.shape[1] / 2 + 10, check.shape[1] / 2 + 10]
#
#     selectionMatrix = np.zeros(check.shape,np.uint8)
#     selectionMatrix[int(selectionFrom[0]):int(selectionTo[0]), int(selectionFrom[1]):int(selectionTo[1])] = 1
#
#     center = searchTemplateCenterPointIn(checkMap, binary, selectionMatrix)
#
#     binaryRadios = [binary.shape[0] / 2, binary.shape[1] / 2]
#
#     binary[binary == 1] = 120
#
#     for i in range(center[0]-binaryRadios[0]-1, center[0]+binaryRadios[0]):
#         for j in range(center[1]-binaryRadios[1], center[1]+binaryRadios[1]):
#             index = [i-center[0]+binaryRadios[0]-1,j-center[1]+binaryRadios[1]]
#             if binary[index[0], index[1]] == 120:
#                 check[i,j] = 120
#     # check[center[0]-binaryRadios[0]-1:center[0]+binaryRadios[0], center[1]-binaryRadios[1]:center[1]+binaryRadios[1]] = binary
#
#     cv2.imwrite('inputFields/checkRes' + name.__str__() + '.png', check)
#
#
# checkTemplate(1)
# checkTemplate(2)
# checkTemplate(3)
# checkTemplate(4)
# checkTemplate(5)
# checkTemplate(6)
# checkTemplate(7)
# checkTemplate(8)

# print img_template
