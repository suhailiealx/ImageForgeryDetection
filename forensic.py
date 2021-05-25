import numpy as np
import numpy.matlib as npm
import argparse
import json
import pprint
import exifread
import cv2 as cv
import os
import pywt
import math
import progressbar
import warnings
from scipy import ndimage
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from matplotlib import pyplot as plt
from os.path import basename


def main():
    argparser = argparse.ArgumentParser(description="Digital Image Forensics")

    argparser.add_argument("datafile", metavar='file', help='name of the image file')
    argparser.add_argument("-e", "--exif", help="forensic with EXIF metadata", action="store_true")
    argparser.add_argument("-gm", "--jpegghostm", help="forensic with JPEG Ghost (Multiple)", action="store_true")
    argparser.add_argument("-g", "--jpegghost", help="forensic with JPEG Ghost", action="store_true")
    argparser.add_argument("-n", "--noise1", help="forensic with noise inconsistencies", action="store_true")
    argparser.add_argument("-mn", "--noise2", help="forensic with Median-filter noise residue inconsistencies", action="store_true")
    argparser.add_argument("-el", "--ela", help="forensic with Error Level Analysis", action="store_true")
    argparser.add_argument("-q", "--quality", help="resaved image quality", type=int)

    # Parses arguments
    args = argparser.parse_args()

    if check_file(args.datafile) == False:
        print("Invalid file. Please make sure the file is exist and the type is JPEG")
        return

    # forensic with exif metadata
    if args.exif:
        exif_check(args.datafile)

    # forensic with jpeg ghost
    elif args.jpegghostm:
        jpeg_ghost_multiple(args.datafile)
    elif args.jpegghost:
        jpeg_ghost(args.datafile, args.quality)

    # forensic with noise inconsitencies
    elif args.noise1:
        noise_inconsistencies(args.datafile)

    # forensic with median filter noise
    elif args.noise2:
        median_noise_inconsistencies(args.datafile)

    # forensic with error level analysis
    elif args.ela:
        ela(args.datafile, args.quality)
    else:
        exif_check(args.datafile)


def check_file(data_path):
    if os.path.isfile(data_path) == False:
        return False
    if data_path.lower().endswith(('.jpg', '.jpeg')) == False:
        return False
    return True


def exif_check(file_path):
    # Open image file
    f = open(file_path, 'rb')

    tags = exifread.process_file(f)

    # Get the pure EXIF data of Image
    exif_code_form = extract_pure_exif(file_path)
    if exif_code_form == None:
        print("The EXIF data has been stripped. Photo maybe is taken from facebook, twitter, imgur")
        return

    check_software_modify(exif_code_form)
    check_modify_date(exif_code_form)
    check_original_date(exif_code_form)
    check_camera_information(tags)

    # Print metadata
    print("\nRAW IMAGE METADATA")
    print("============================================================= \n")
    print("EXIF Data")

    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            print("%-35s:  %s" % (tag, tags[tag]))

def extract_pure_exif(file_name):
    img = Image.open(file_name)
    info = img._getexif()
    return info

def decode_exif_data(info):
    exif_data = {}
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exif_data[decoded] = value

    return exif_data

def get_if_exist(data, key):
    if key in data:
        return data[key]
    return None

def export_json(data):
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

# Check software edit
def check_software_modify(info):
    software = get_if_exist(info, 0x0131)
    if software != None:
        print("Image edited with: %s" % software)
        return True
    return False

# Check modify date
def check_modify_date(info):
    modify_date = get_if_exist(info, 0x0132)
    if modify_date != None:
        print("Photo has been modified since it was created. Modified: %s" % modify_date)
        return True
    return False

# Check original date
def check_original_date(info):
    original_date = get_if_exist(info, 0x9003)
    create_date = get_if_exist(info, 0x9004)
    if original_date != None:
        print("The original date is: %s" % original_date)
    if create_date != None:
        print("Image created at: %s" % create_date)

# Check camera information
def check_camera_information_2(info):
    make = get_if_exist(info, 0x010f)
    model = get_if_exist(info, 0x0110)
    exposure = get_if_exist(info, 0x829a)
    aperture = get_if_exist(info, 0x829d)
    focal_length = get_if_exist(info, 0x920a)
    iso_speed = get_if_exist(info, 0x8827)
    flash = get_if_exist(info, 0x9209)

    print("\nCamera Infomation")
    print("Make: \t \t %s" % make)
    print("Model: \t \t %s" % model)
    print("ISO Speed: \t %s" % iso_speed)
    print("Flash: \t \t %s" % flash)

def check_camera_information(info):
    make = get_if_exist(info, 'Image Make')
    model = get_if_exist(info, 'Image Model')
    exposure = get_if_exist(info, 'EXIF ExposureTime')
    aperture = get_if_exist(info, 'EXIF ApertureValue')
    focal_length = get_if_exist(info, 'EXIF FocalLength')
    iso_speed = get_if_exist(info, 'EXIF ISOSpeedRatings')
    flash = get_if_exist(info, 'EXIF Flash')

    print("\nCamera Infomation")
    print("-------------------------------------------------------------- ")
    print("Make: \t \t %s" % make)
    print("Model: \t \t %s" % model)
    print("Exposure: \t %s " % exposure)
    print("Aperture: \t %s" % aperture)
    print("Focal Length: \t %s mm" % focal_length)
    print("ISO Speed: \t %s" % iso_speed)
    print("Flash: \t \t %s" % flash)

def jpeg_ghost_multiple(file_path):
    print("Analyzing...")

    bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    img = cv.imread(file_path)
    img_rgb = img[:, :, ::-1]

    quality = 60

    smoothing_b = 17
    offset = int((smoothing_b-1)/2)

    height, width, channels = img.shape

    # Plot the original image
    plt.subplot(5, 4, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.xticks([]), plt.yticks([])

    base = basename(file_path)
    file_name = os.path.splitext(base)[0]
    save_file_name = file_name+"_temp.jpg"
    bar.update(1)

    # Try 19 different qualities
    for pos_q in range(19):

        # Resaved the image with the new quality
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(save_file_name, img, encode_param)

        # Load resaved image
        img_low = cv.imread(save_file_name)
        img_low_rgb = img_low[:, :, ::-1]

        # Compute the square different between original image and the resaved image
        tmp = (img_rgb-img_low_rgb)**2

        # Take the average by kernel size b
        kernel = np.ones((smoothing_b, smoothing_b),
                         np.float32)/(smoothing_b**2)
        tmp = cv.filter2D(tmp, -1, kernel)

        # Take the average of 3 channels
        tmp = np.average(tmp, axis=-1)

        # Shift the pixel from the center of the block to the left-top
        tmp = tmp[offset:(int(height-offset)), offset:(int(width-offset))]

        # Compute the nomalized component
        nomalized = tmp.min()/(tmp.max() - tmp.min())

        # Nomalization
        dst = tmp - nomalized

        # Plot the diffrent images
        plt.subplot(5, 4, pos_q+2), plt.imshow(dst, cmap='gray'), plt.title(quality)
        plt.xticks([]), plt.yticks([])
        quality = quality + 2
        bar.update(pos_q+2)

    bar.finish()
    print("Done")
    plt.suptitle('Forensic with JPEG Ghost')
    plt.show()
    os.remove(save_file_name)


def jpeg_ghost(file_path, quality):

    print("Analyzing...")
    bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    img = cv.imread(file_path)
    img_rgb = img[:, :, ::-1]

    # Quality of the reasaved images
    if quality == None:
        quality = 60

    # Size of the block
    smoothing_b = 17
    offset = (smoothing_b-1)/2

    # Size of the image
    height, width, channels = img.shape

    # Plot the original image
    plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title('Image')
    plt.xticks([]), plt.yticks([])

    # Get the name of the image
    base = basename(file_path)
    file_name = os.path.splitext(base)[0]
    save_file_name = file_name+"_temp.jpg"
    bar.update(1)

    # Resaved the image with the new quality
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    cv.imwrite(save_file_name, img, encode_param)

    # Load resaved image
    img_low = cv.imread(save_file_name)
    img_low_rgb = img_low[:, :, ::-1]
    bar.update(5)
    # Compute the square different between original image and the resaved image
    tmp = (img_rgb-img_low_rgb)**2

    # Take the average by kernel size b
    kernel = np.ones((smoothing_b, smoothing_b), np.float32)/(smoothing_b**2)
    tmp = cv.filter2D(tmp, -1, kernel)
    bar.update(10)
    # Take the average of 3 channels
    tmp = np.average(tmp, axis=-1)

    # Shift the pixel from the center of the block to the left-top
    tmp = tmp[int(offset):int(height-offset), int(offset):int(width-offset)]

    # Compute the nomalized component
    nomalized = tmp.min()/(tmp.max() - tmp.min())
    bar.update(15)
    # Nomalization
    dst = tmp - nomalized

    # Plot the diffrent images
    plt.subplot(1, 2, 2), plt.imshow(dst), plt.title("Analysis. Quality = " + str(quality))
    plt.xticks([]), plt.yticks([])
    bar.update(20)

    bar.finish()
    print("Done")
    plt.suptitle('forensic with JPEG Ghost')
    plt.show()
    os.remove(save_file_name)

def noise_inconsistencies(file_path, block_size=8):

    print("Analyzing...")
    bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    img = cv.imread(file_path)
    img_rgb = img[:, :, ::-1]

    imgYCC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, _, _ = cv.split(imgYCC)

    coeffs = pywt.dwt2(y, 'db8')
    bar.update(5)

    cA, (cH, cV, cD) = coeffs
    cD = cD[0:(len(cD)//block_size)*block_size, 0:(len(cD[0])//block_size)*block_size]
    block = np.zeros((len(cD)//block_size, len(cD[0])//block_size, block_size**2))
    bar.update(10)

    for i in range(0, len(cD), block_size):
        for j in range(0, len(cD[0]), block_size):
            blockElement = cD[i:i+block_size, j:j+block_size]
            temp = np.reshape(blockElement, (1, 1, block_size**2))
            block[int((i-1)/(block_size+1)),
                  int((j-1)/(block_size+1)), :] = temp

    bar.update(15)
    abs_map = np.absolute(block)
    med_map = np.median(abs_map, axis=2)
    noise_map = np.divide(med_map, 0.6745)
    bar.update(20)

    bar.finish()
    print("Done")

    plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title('Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(noise_map), plt.title('Analysis')
    plt.xticks([]), plt.yticks([])
    plt.suptitle('Forensic using noise inconsistencies')
    plt.show()


def median_noise_inconsistencies(file_path, n_size=3):
    print("Analyzing...")
    bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    img = cv.imread(file_path)
    img_rgb = img[:, :, ::-1]

    flatten = True
    multiplier = 10
    bar.update(5)

    img_filtered = img

    img_filtered = cv.medianBlur(img, n_size)

    noise_map = np.multiply(np.absolute(img - img_filtered), multiplier)
    bar.update(15)

    if flatten == True:
        noise_map = cv.cvtColor(noise_map, cv.COLOR_BGR2GRAY)
    bar.update(20)
    bar.finish()
    print("Done")

    plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title('Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(noise_map), plt.title('Analysis')
    plt.xticks([]), plt.yticks([])
    plt.suptitle('forensic with Median-filter noise residue inconsistencies')
    plt.show()


def ela(file_path, quality, block_size=8):
    print("Analyzing...")
    bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    img = cv.imread(file_path)
    img_rgb = img[:, :, ::-1]
    bar.update(5)

    # Get the name of the image
    base = basename(file_path)
    file_name = os.path.splitext(base)[0]
    save_file_name = file_name+"_temp.jpg"

    if quality == None:
        quality = 90
    multiplier = 15
    flatten = True

    # Resaved the image with the new quality
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    cv.imwrite(save_file_name, img, encode_param)
    bar.update(10)

    # Load resaved image
    img_low = cv.imread(save_file_name)
    img_low = img_low[:, :, ::-1]

    ela_map = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3))

    ela_map = np.absolute(1.0*img_rgb - 1.0*img_low)*multiplier

    bar.update(15)
    if flatten == True:
        ela_map = np.average(ela_map, axis=-1)
    bar.update(20)
    bar.finish()
    print("Done")

    plt.subplot(1, 2, 1), plt.imshow(img_rgb), plt.title('Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(ela_map), plt.title('Analysis')
    plt.xticks([]), plt.yticks([])
    plt.suptitle('forensic with Error Level Analysis')
    plt.show()
    os.remove(save_file_name)

if __name__ == "__main__":
    main()
