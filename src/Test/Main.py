'''
Created on 26 de jun de 2019

@author: zeller
'''

#===============================================================================
#                                     Imports
#===============================================================================
from DIP.Image import *
from pathlib import Path
from posix import listdir
#===============================================================================
#                                     Main
#===============================================================================
def absolute_path_workspace ():
    return Path(__file__).parent.parent.parent

def resolve_path (relative_path):
    return absolute_path_workspace() / relative_path

def get_images(operation_path):
    base_path = resolve_path('assets/images' +  operation_path)
    files = listdir(base_path)
    return [load_image_path(base_path / path) for path in files]

def get_images_to_operation (operation_path):
    return get_images(operation_path)

def show_images (images):
    for index, image in enumerate(images):
        image.show(f'Images - image_{index}')

def get_images_to_k_means ():
    return get_images_to_operation('/k_means')

def get_license_plate_images ():
    return get_images_to_operation('/k_means')

def add_images_demo ():
    images = get_images_to_operation('/add')
    show_images(images)
    add_images(images, True).show('Add images and Normalize Result')

def sub_images_demo ():
    images = get_images_to_operation('/sub')
    images.reverse()
    show_images(images)
    subtract_images(images, True).show('Subtract images and Normalize Result')

def multiply_images_demo ():
    images = get_images_to_operation('/sub')
    show_images(images)
    multiply_images(images, True).show('Multiply Images and Normalize Result')

def half_median_filter_demo ():
    images = get_images_to_operation('/half_median')
    for image in images: 
        image.show('Before Filter')
        half_median_filter(image).show('Laplacian Filter')

def median_filter_demo ():
    images = get_images_to_operation('/half_median')
    for image in images: 
        image.show('Before Filter')
        median_filter(image).show('Median Filter')
        
def laplacian_kernel_demo ():
    images = get_images_to_operation('/laplacian')
    for image in images: 
        image.show('Before Filter')
        laplacian_filter(image).show('Laplacian Filter')

def automatic_threshold_image_demo():
    images = get_images_to_operation('/threshold')
    for image in images: 
        image.show('Before Threshold')
        threshold_value = find_threshold_value(image, START_IMAGE_HEIGHT, START_IMAGE_WIDTH, image.height, image.width)
        threshold_image(image, START_IMAGE_HEIGHT, START_IMAGE_WIDTH, image.height, image.width, threshold_value).show('Automatic Threshold Filter')

def manual_threshold_image_demo():
    images = get_images_to_operation('/threshold')
    for image in images: 
        image.show('Before Threshold')
        threshold_value = 126
        threshold_image(image, START_IMAGE_HEIGHT, START_IMAGE_WIDTH, image.height, image.width, threshold_value).show('Manual Threshold Filter')

def key_means_demo ():
    images = get_images_to_operation('/k_means')
    for image in images: 
        image.show('Before K-Means')
        cluster_by_k_means_method(image, 3).show('K-Means Filter with 3 Clusters')

def find_license_plates_demo ():
    images = get_images_to_operation('/license_plate')
    for image in images:
        base_image = image
        image = luminosity_monocromatization(image)
        image.show("Base Image")
        image = threshold_image(image, START_IMAGE_HEIGHT, START_IMAGE_WIDTH, image.height, image.width, 90)
        image = sobel_filter(image)
        image.show("Sobel Filter")
        sobel_histogram_values = generate_sobel_histogram(image)
        license_plate_areas = find_license_plate_areas(sobel_histogram_values)
        show_license_plate_areas(image, license_plate_areas).show("License Plate Areas")
        create_sobel_histogram(sobel_histogram_values).show("Sobel Histogram")
        license_plate_location = find_license_plate(license_plate_areas, sobel_histogram_values)
        start_position = license_plate_location[0] - int(license_plate_location[0] * 0.07) if license_plate_location[0] - int(license_plate_location[0] * 0.07) >= 0 else license_plate_location[0] 
        end_position = license_plate_location[1]
        find_license_plates(image, start_position, end_position).show("Modified Image")
        find_license_plates(base_image, start_position, end_position).show("Base Image")

def run_demo ():
    # add_images_demo()
    # sub_images_demo()
    # multiply_images_demo()
    # half_median_filter_demo()
    # median_filter_demo()
    # laplacian_kernel_demo()
    # manual_threshold_image_demo()
    # automatic_threshold_image_demo()
    # key_means_demo()
    find_license_plates_demo()

if __name__ == '__main__':
    run_demo() 
