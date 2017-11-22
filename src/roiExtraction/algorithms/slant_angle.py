# ---Libraries---
# Standard library
import os
import sys

# Third-party libraries
import cv2
import numpy as np

# Private libraries
sys.path.append(os.path.abspath("../"))
import utils

# ---About---
# This file contain slant angle extraction or matrix slant angle extraction function.


#       calculate_slant_angle:
#         Input:
#           - image, binary image
#           - threshold, which means minimum vote it should get for it to be considered as a line
#           - axis, image axis for the angle histogram
#           - minDegree, minimum degree to by evaluate in the algorithm
#           - maxDegree, maximum degree to by evaluate in the algorithm
#         Output:
#           - angle, the average angle that was sampled from the lines that was detected in the image
def calculate_slant_angle(image, threshold=5, axis=1, minDegree=-10, maxDegree=10):
    angleMatrix = calculate_slant_angle_matrix(image, threshold, axis, minDegree, maxDegree)

    return utils.angle_matrix_average(angleMatrix)


#       calculate_slant_angle_pp:
#           calculate image slant using projection profile algorithm,
#           the algorithm calculate image histogram in range of angles and
#           return The angle at which the histogram gives the most peaks by maximize std function.
#
#         Input:
#           - image, binary image
#           - axis, image axis for the angle histogram
#           - minDegree, minimum degree to by evaluate in the algorithm
#           - maxDegree, maximum degree to by evaluate in the algorithm
#           - step, step angle
#         Output:
#           - angle, the angle that will fix the image
def calculate_slant_angle_pp(image, axis=1, minDegree=-7, maxDegree=7, step=0.5, isGrayImage=False):
    maxMargin = 0
    slantAngle = 0

    for currAngle in np.arange(minDegree, maxDegree, step):
        hist = utils.image_histogram(image, rotationAngle=currAngle, axis=axis, isGrayImage=isGrayImage)

        margin = np.std(hist)

        if margin > maxMargin:
            maxMargin = margin
            slantAngle = currAngle

    return slantAngle


def fined_lines_in_check(image, fileName, minLineLength=200, maxLineGap=0):
    image = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (21, 21), 0)
    ret3, invers = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("OTSU" + fileName, invers)
    ret, invers = cv2.threshold(invers, 127, 255, cv2.THRESH_BINARY_INV)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imwrite("invers" + fileName, invers)

    kernel = np.ones((9, 9), np.float32)
    # kernel[:, 4]
    kernel[0,:] = -0.5
    kernel[8, :] = -0.5
    kernel[1, :] = -0.5
    kernel[7, :] = -0.5
    kernel[2, :] = -0.5
    kernel[6, :] = -0.5
    kernel[5,5] = 0
    kernel[5,4] = 0
    kernel[5,6] = 0
    kernel[4,5] = 0
    kernel[4,4] = 0
    kernel[4,6] = 0
    kernel[6,5] = 0
    kernel[6,4] = 0
    kernel[6,6] = 0


    # kernel *= 2
    dst = cv2.filter2D(invers, -1, kernel)
    blur = cv2.GaussianBlur(invers, (5, 5), 0)
    cv2.imwrite("kernel" + fileName, blur)

    lines = cv2.HoughLinesP(blur, 1, np.pi / 180, 200, minLineLength, maxLineGap)

    lineCurr = 0
    [[x1, y1, x2, y2]] = lines[lineCurr]
    line = [x1, y1, x2, y2]

    while utils.line_length_is_grater_then(0.005*image.shape[1], line):
        if -1 < utils.get_angle(x1, y1, x2, y2) < 1:
            cv2.line(invers, (x1, y1), (x2, y2), (255, 255, 255), 3)

        lineCurr += 1
        [[x1, y1, x2, y2]] = lines[lineCurr]
        line = [x1, y1, x2, y2]

    cv2.imwrite("dst2" + fileName, dst)

    lines = cv2.HoughLinesP(invers, 1, np.pi / 180, 200, minLineLength, maxLineGap)

    lineCurr = 0
    [[x1, y1, x2, y2]] = lines[lineCurr]
    line = [x1, y1, x2, y2]

    while utils.line_length_is_grater_then(0.05 * image.shape[1], line):
        if -1 < utils.get_angle(x1, y1, x2, y2) < 1:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        lineCurr += 1
        [[x1, y1, x2, y2]] = lines[lineCurr]
        line = [x1, y1, x2, y2]


    print lineCurr
    cv2.imwrite(fileName, image)


#       Slant angle matrix:
#         Input:
#           - image, binary image
#           - threshold, which means minimum vote it should get for it to be considered as a line
#           - axis, image axis for the angle histogram
#           - minDegree, minimum degree to by evaluate in the algorithm
#           - maxDegree, maximum degree to by evaluate in the algorithm
#         Output:
#           - angle_matrix,  matrix of image [axis] on angle (0-360) histogram
def calculate_slant_angle_matrix(image, threshold=5, axis=1, minDegree=5, maxDegree=24):
    angleMatrix = np.zeros((image.shape[axis], 360))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold)

    for line_s in lines:
        for rho, theta in line_s:
            angle = int(np.rad2deg(theta))
            if (minDegree < angle < maxDegree or
                            minDegree < 180 - angle < maxDegree):
                a = np.cos(theta)
                b = np.sin(theta)

                if axis == 1:
                    x = int((rho - image.shape[0] / 2 * b) / a)
                    index = x
                else:
                    y = int((rho - image.shape[1] / 2 * a) / b)
                    index = y

                if 0 < index < image.shape[axis]:
                    angleMatrix[index, angle] += 1

                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))

                cv2.line(image, (x1, y1), (image.shape[1], y), (0, 0, int(theta * 255 / np.pi)), 2)

    cv2.imwrite('test.png', image)
    return angleMatrix


#       Fix check image:
#         Input:
#           - path, path for bank check image
#         Output:
#           - image,  fix image of the check
def fix_check(path):
    _check_img = cv2.imread(path)
    _angle = calculate_slant_angle_pp(_check_img)
    _fix_check_img = utils.rotate_image_by_angle(_check_img, _angle)

    _fix_check_img = np.array(_fix_check_img, dtype=np.uint8)
    _gray = cv2.cvtColor(_fix_check_img, cv2.COLOR_BGR2GRAY)
    ret3, _fix_binary_check_img = cv2.threshold(_gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return _fix_binary_check_img




# def amount_extraction(check_img):
#     amount = (int(0.74 * check_img.shape[1]),
#               int(0.48 * check_img.shape[0]),
#               int(0.96 * check_img.shape[1]),
#               int(0.58 * check_img.shape[0]))
#     img2 = check_img[amount[1]:amount[3], amount[0]:amount[2]]
#     return img2
#
#
# def date_extraction(check_img):
#     amount = (int(0.45 * check_img.shape[1]),
#               int(0.62 * check_img.shape[0]),
#               int(0.71 * check_img.shape[1]),
#               int(0.72 * check_img.shape[0]))
#     img2 = check_img[amount[1]:amount[3], amount[0]:amount[2]]
#     return img2
#
#
# def date_extraction(check_img):
#     amount = (int(0.45 * check_img.shape[1]),
#               int(0.62 * check_img.shape[0]),
#               int(0.71 * check_img.shape[1]),
#               int(0.72 * check_img.shape[0]))
#     img2 = check_img[amount[1]:amount[3], amount[0]:amount[2]]
#     return img2



# ---Example and test case---
# img_matrix = cv2.imread('../../data/cvl.str/25000-0001-08.png')
# angle_matrix = slant_angle_matrix(img_matrix)
# print np.array_equal(angle_matrix, np.zeros((img_matrix.shape[1], 360)))
#
# img_angle = cv2.imread('../../assets/Checks/1.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test1.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif1.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif1.png',color_BIFs.bifs_to_color_image(obifs))
# print angle == -2 #-2.213
#
# img_angle = cv2.imread('../../assets/Checks/2.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test2.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif2.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif2.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test2.png', image2)
# print angle == -2.5 #-2.423
#
# img_angle = cv2.imread('../../assets/Checks/3.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test3.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif3.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif3.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test3.png', image2)
# print angle == -1 #-1.054
#
# img_angle = cv2.imread('../../assets/Checks/4.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test4.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif4.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif4.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test4.png', image2)
# print angle == 0 #-0.969
#
# img_angle = cv2.imread('../../assets/Checks/5.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test5.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif5.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif5.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test5.png', image2)
# print angle == 0 #-0.138
#
# img_angle = cv2.imread('../../assets/Checks/6.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test6.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif6.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif6.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test6.png', image2)
# print angle == 1 #0.968
#
# img_angle = cv2.imread('../../assets/Checks/7.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test7.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif7.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif7.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test7.png', image2)
# print angle == 0 #-0.549
#
# img_angle = cv2.imread('../../assets/Checks/8.png')
# angle = calculate_slant_angle_pp(img_angle)
# image2 = utils.rotate_image_by_angle(img_angle, angle)
# # fined_lines_in_check(image2, 'test8.png')
# [bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
# cv2.imwrite('bif8.png', color_BIFs.bifs_to_color_image(bifs))
# obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
# cv2.imwrite('Obif8.png',color_BIFs.bifs_to_color_image(obifs))
# # cv2.imwrite('test8.png', image2)
# print angle == 1.5 #1.317


# image2 = utils.rotate_image_by_angle(img_angle, angle - 90)
# cv2.imwrite('test1.png', image2)

# amount_image = amount_extraction(image2)
# cv2.imwrite('amount.png', amount_image)
#
#
# date_image = date_extraction(image2)
# cv2.imwrite('date.png', date_image)


# ---Internal function---
