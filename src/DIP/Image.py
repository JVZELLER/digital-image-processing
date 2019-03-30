'''
Created on 24 de mar de 2019

@author: zeller
'''
#===============================================================================
#                                     Imports
#===============================================================================
# To work with images
from PIL import Image
# To make HTTP requests
import requests
# To open images as bytes
from io import BytesIO
# To display images
from IPython.display import display
# To create matrix from images
import numpy as np

#===============================================================================
#                                    Constants
#===============================================================================
BANDS = 3
MODE = 'RGB'
DEFAULT_MATRIX_TYPE = 'uint8'
DEFAULT_RBG_VALUE = ( 0, 0, 0 )
DEFAULT_PIXEL_VALUE = MAX_PIXEL_VALUE = 255
MIN_PIXEL_INDEX, MAX_PIXEL_INDEX = 0, 1
R, G, B = 0, 1, 2
SIZE = ( 200, 300 )
PUND_LUMINOSITY_MODES = {
    "BT709": [0.2125, 0.7154, 0.0721],
    "RMY": [0.5, 0.419, 0.081],
    "Y": [0.299, 0.587, 0.114]
    }


#===============================================================================
#                                 Image Manipulation
#===============================================================================
def load_image_url ( url ):
    """Carrega imagem a partir de url
    Args:
        url: url da imagem a ser carregada
    Returns:
        Um novo objeto imagem (Pillow.Image)
        
    """
    imagem_remota = requests.get( url )
    return Image.open( BytesIO( imagem_remota.content ) )


def load_image_path ( path ):
    """Carrega imagem a partir de path relativo
    Args:
        path: path da imagem a ser carregada
    Returns:
        Um novo objeto imagem (Pillow Image)
    """
    return Image.open( path )


def new_image ( image_size = SIZE, color_mode = MODE, pixel_rgb_value = DEFAULT_RBG_VALUE ):
    """Cria nova imagem (Pillow.Image)
    Args:
        color_mode: esquema de cores da imagem
        image_size: tupla contendo o width e higth da imagem. Default = ( 200, 300 )
        pixel_rbg_value: valor de cada pixel nos canais R, G e B
    Returns:
        Um novo objeto imagem (Pillow image)
    """
    return Image.new( color_mode, image_size, pixel_rgb_value )


def load_image_data ( image ):
    """Carrega dados da imagem
    Args:
        image: imagem com dados a serem carregados
    Returns:
        Um novo objeto imagem
    """
    return image.load()


def matrix_from_image( lines, columns, bands = BANDS, pixel_value = DEFAULT_PIXEL_VALUE ):
    """Cria matriz de pixels
    Args:
        lines: número de linhas da matriz (height)
        columns: número de colunas da matriz (width)
        bands: número de bandas (dimensoes) da imagem. Default = 3 (BANDS)
        pixel_value: valor atribuido a cada pixel da imagem. Default = 255 (DEFAULT_PIXEL_VALUE)
    Returns:
        Matriz com n=bands dimensoes de pixels
    """
    return [ [ [ band * 0 + pixel_value for band in range( bands ) ] for column in range( columns ) ] for line in range( lines ) ]


def image_from_matrix( rgb_matrix, matrix_Type = DEFAULT_MATRIX_TYPE, color_mode = MODE ):
    """Cria imagem a partir de matriz
    Args:
        rgb_matrix: matriz tridimensional representando imagem
        canais R, G e B.
        matrix_Type: tipo de dados da matriz, default = uint8
        mode: image color scheme. Default = 'RGB'
    Returns:
        new Pillow Image
    """
    return Image.fromarray( np.asarray( rgb_matrix ).astype( matrix_Type ), color_mode )


def find_min_image_height ( images ):
    """Retorna menor altura dentre lista de imagens
    Args:
        images: lista de imagens
    Returns:
        menor altura
    """
    return min( image.height for image in images )


def find_min_image_width( images ):
    """Retorna menor largura dentre lista de imagens
    Args:
        images: lista de imagens
    Returns:
        menor largura
    """
    return min( image.width for image in images )


def new_kernel ( lines = 3, columns = 3 ):
    """Cria kernel a partir de lines e columns (linhas e colunas)
    Args:
        lines: número de linhas do kernel, default = 3
        columns: número de colunas do kernel, default = 3
    Returns:
        kernel (lines x columns)
    """
    return [ [ line * 0 + 1 for line in range( lines ) ] for column in range( columns ) ]


def find_kernel_width( kernel ):
    """Descobre largura do kernel (width)
    Args:
        kernel: kernel que se deseja-se descobrir largura
    Returns:
        largura do kernel
    """
    return len( kernel )


def find_kernel_height( kernel ):
    """Descobre altura do kernel (width)
    Args:
        kernel: kernel que se deseja-se descobrir altura
    Returns:
        altura do kernel
    """
    return len( kernel[0] )


def start_kernel_width ( kernel ):
    """Descobre largura máxima para excursão do kernel
    Args:
        kernel: kernel que se deseja-se descobrir largura
        máxima para movimentação (excursão)
    Returns:
        largura máxima para excursão
    """
    kernel_width = find_kernel_width( kernel )
    
    return kernel_width // 2


def start_kernel_height ( kernel ):
    """Descobre altura máxima para excursão do kernel
    Args:
        kernel: kernel que se deseja-se descobrir altura
        máxima para movimentação (excursão)
    Returns:
        altura máxima para excursão
    """
    kernel_height = find_kernel_height( kernel )
    
    return kernel_height // 2

def get_image_width ( image ):
    return len( image )

def get_image_height ( image ):
    return len( image[0] )


#===============================================================================
#                                DIP Algorithms
#===============================================================================
def find_rgb_image_bounds( matrix_image_data, matrix_image_width, matrix_image_height ):
    """Econtra maior e menor valor de pixel dada uma imagem
    Args:
        image: matriz da image
        matrix_image_width: largura da matriz da imagem
        matrix_image_height: altura da matriz da imagem
    Returns: lista contendo maior e menor valor de cada canal
    """
    min_pixel_value = None
    max_pixel_value = None
    image_bounds = [[], [], []]
    
    for band in range ( BANDS ):
        for position_x in range( matrix_image_width ):
            for position_y in range( matrix_image_height ):
                if min_pixel_value == None or matrix_image_data[position_x][position_y][band] < min_pixel_value:
                    min_pixel_value = matrix_image_data[position_x][position_y][band]
                if max_pixel_value == None or matrix_image_data[position_x][position_y][band] > max_pixel_value:
                    max_pixel_value = matrix_image_data[position_x][position_y][band]
                        
        image_bounds[band] = [ min_pixel_value, max_pixel_value ]
        
    return image_bounds
            

def normalize_rgb_image ( matrix_image ):
    """Normaliza imagem utilzando regra de três simples
    Args:
        matrix_image: matriz de imagem RGB
    Returns: matriz de imagem RGB com valores dos pixels normalizados
    """    
    image_width = get_image_width( matrix_image )
    image_height = get_image_height( matrix_image )
    
    imageBound = find_rgb_image_bounds( matrix_image, image_width, image_height )
    
    fator_ajuste_r = 255 / ( imageBound[R][MAX_PIXEL_INDEX] - imageBound[R][MIN_PIXEL_INDEX] )
    fator_ajuste_g = 255 / ( imageBound[G][MAX_PIXEL_INDEX] - imageBound[G][MIN_PIXEL_INDEX] )
    fator_ajuste_b = 255 / ( imageBound[B][MAX_PIXEL_INDEX] - imageBound[B][MIN_PIXEL_INDEX] )
    
    for position_x in range( image_width ):
        for position_y in range( image_height ):
            matrix_image[position_x][position_y] = ( 
                            ( fator_ajuste_r * ( matrix_image[position_x][position_y][R] - imageBound[R][0] ) ),
                            ( fator_ajuste_g * ( matrix_image[position_x][position_y][G] - imageBound[G][0] ) ),
                            ( fator_ajuste_b * ( matrix_image[position_x][position_y][B] - imageBound[B][0] ) )
                          )
    
    return matrix_image


def add_images( images, normalize_result = False, color_mode = MODE ):
    """Soma N imagens tratando overflow com truncamento ou normalização
    Args: 
        images: lista de imagens
        normalize_result: indica truncamento(False) ou normalização(True), default=False
        color_mode = 'color color_mode' da imagem resultante, defaul='RGB'
    Returns:
        Um objeto de imagem contendo a soma de todas as 
        imagens.
    """
    new_image_width = find_min_image_width( images )
    new_image_height = find_min_image_height( images )
    
    base_matrix_image = matrix_from_image( new_image_height, new_image_width )
    
    for image in range( len( images ) - 1 ):
        image1 = load_image_data( images[image] )
        image2 = load_image_data( images[image + 1] )
        for position_x in range( new_image_width ):
            for position_y in range( new_image_height ):
                base_matrix_image[position_y][position_x] = ( 
                                   image1[position_x, position_y][R] + image2[position_x, position_y][R],
                                   image1[position_x, position_y][G] + image2[position_x, position_y][G],
                                   image1[position_x, position_y][B] + image2[position_x, position_y][B]
                                  )
    if( normalize_result ):
        base_matrix_image = normalize_rgb_image( base_matrix_image )
        
    return image_from_matrix( base_matrix_image )


def subtract_images( images, normalize_result = False, color_mode = MODE ):
    """Subtrai N imagens tratando overflow com truncamento ou normalizacao
    Args: 
        images: lista de imagens
        normalize_result: indica truncamento(False) ou normalização(True), default=False
        color_mode = 'color color_mode' da imagem resultante, defaul='RGB'
    Returns:
        Um objeto de imagem contendo a subtração de todas as 
        imagens.
    """
    new_image_width = find_min_image_width( images )
    new_image_height = find_min_image_height( images )
    
    baseImage = matrix_from_image( new_image_height, new_image_width )
    
    for image in range( len( images ) - 1 ):
        image1 = load_image_data( images[image] )
        image2 = load_image_data( images[image + 1] )
        for position_x in range( new_image_width ):
            for position_y in range( new_image_height ):
                baseImage[position_y][position_x] = ( 
                                   image1[position_x, position_y][R] - image2[position_x, position_y][R],
                                   image1[position_x, position_y][G] - image2[position_x, position_y][G],
                                   image1[position_x, position_y][B] - image2[position_x, position_y][B]
                                  )
    if( normalize_result ):
        baseImage = normalize_rgb_image( baseImage )
        
    return image_from_matrix( baseImage )


def multiply_images( images, normalize_result = False, color_mode = MODE ):
    """Multiplica N imagens
    Args: 
        images: lista de imagens
        normalize_result: indica truncamento(False) ou normalização(True), default=False
        color_mode = 'color color_mode' da imagem resultante, defaul='RGB'
    Returns:
        Um objeto de imagem contendo a multiplicação de todas as 
        imagens.
    """
    new_image_width = find_min_image_width( images )
    new_image_height = find_min_image_height( images )
    
    baseImage = matrix_from_image( new_image_height, new_image_width )
    
    for image in range( len( images ) - 1 ):
        image1 = load_image_data( images[image] )
        image2 = load_image_data( images[image + 1] )
        for position_x in range( new_image_width ):
            for position_y in range( new_image_height ):
                baseImage[position_y][position_x] = ( 
                                   image1[position_x, position_y][R] * image2[position_x, position_y][R],
                                   image1[position_x, position_y][G] * image2[position_x, position_y][G],
                                   image1[position_x, position_y][B] * image2[position_x, position_y][B]
                                  )
    if( normalize_result ):
        baseImage = normalize_rgb_image( baseImage )
        
    return image_from_matrix( baseImage )


def median_filter ( image, kernel = new_kernel( 3, 3 ) ):
    
    resultImage = matrix_from_image( image.height, image.width )
    baseImage = load_image_data( image )
    
    for image_position_x in range( image.width ):
        for image_position_y in range( image.height ):
            
            start_position_x_kernel = 0 if image_position_x - start_kernel_width( kernel ) <= 0 else image_position_x - start_kernel_width( kernel )
            end_position_x_kernel = image_position_x if image_position_x + start_kernel_width( kernel ) >= image.width else image_position_x + ( kernel )
            
            start_position_y = 0 if image_position_y - start_kernel_height( kernel ) <= 0 else image_position_y - start_kernel_height( kernel )
            end_position_y = image_position_y if image_position_y + start_kernel_height( kernel ) >= image.height else image_position_y + start_kernel_height( kernel )
            
            kernelSize = ( end_position_x_kernel + 1 - start_position_x_kernel ) * ( end_position_y + 1 - start_position_y )
            
            for band in range( BANDS ):
                for kernel_position_x in range( start_position_x_kernel, end_position_x_kernel ):
                    kernelX = end_position_x_kernel - start_position_x_kernel
                    for kernel_position_y in range( start_position_y, end_position_y ):
                        kernelY = end_position_y - start_position_y
                        resultImage[image_position_y][image_position_x][band] += baseImage[kernel_position_x, kernel_position_y][band] * kernel[kernelY][kernelX]
                
                resultImage[image_position_y][image_position_x][band] /= kernelSize
            
    return image_from_matrix( resultImage )


def brightness_monocromatization ( image ):
    base_image = load_image_data( image )
    base_matrix_image = image_from_matrix( image.height, image.width )
    
    for image_position_x in range( image.width ):
        for image_position_y in range ( image.height ):
            chanelValues = [
                base_image[image_position_x, image_position_y][R], 
                base_image[image_position_x, image_position_y][G], 
                base_image[image_position_x, image_position_y][B]
            ]
            
            base_matrix_image[image_position_y][image_position_x][R] = ( max( chanelValues ) + min( chanelValues ) ) / 2
            base_matrix_image[image_position_y][image_position_x][G] = ( max( chanelValues ) + min( chanelValues ) ) / 2
            base_matrix_image[image_position_y][image_position_x][B] = ( max( chanelValues ) + min( chanelValues ) ) / 2
    
    return image_from_matrix( base_matrix_image )


def median_monocromatization( image ):
    base_image = load_image_data( image )
    base_matrix_image = image_from_matrix( image.height, image.width )
    
    for image_position_x in range( image.width ):
        for image_position_y in range ( image.height ):
            chanel_values = base_image[image_position_x, image_position_y][R] + base_image[image_position_x, image_position_y][G] + base_image[image_position_x, image_position_y][B]
                            
            base_matrix_image[image_position_y][image_position_x][R] = chanel_values / 3
            base_matrix_image[image_position_y][image_position_x][G] = chanel_values / 3
            base_matrix_image[image_position_y][image_position_x][B] = chanel_values / 3
    
    return image_from_matrix( base_matrix_image )


def luminosity_monocromatization( image, luminosity_mode = "BT709" ):
    chanel_pound = PUND_LUMINOSITY_MODES.get( luminosity_mode )
    base_image = load_image_data( image )
    base_matrix_image = image_from_matrix( image.width, image.height )
    
    for image_position_x in range( image.width ):
        for image_position_y in range ( image.height ):
            chanel_values = [base_image[image_position_x, image_position_y][R], base_image[image_position_x, image_position_y][G], base_image[image_position_x, image_position_y][B]]
            base_matrix_image[image_position_y][image_position_x][R] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
            base_matrix_image[image_position_y][image_position_x][G] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
            base_matrix_image[image_position_y][image_position_x][B] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
    
    return image_from_matrix( base_matrix_image )


def generate_histogram( image ):
    grey_scale_frequency = [i * 0 for i in range( 256 )]
    base_image = load_image_data( image )
    
    for image_position_x in range( image.width ):
        for image_position_y in range( image.height ):
            pixel_value = base_image[image_position_x, image_position_y][R]
            grey_scale_frequency[pixel_value] += 1
    
    image_resolution = image.width * image.height
    max_grey_scale_frequency = max( grey_scale_frequency )

    for index in range( len( grey_scale_frequency ) ):
        grey_scale_frequency[index] = int( ( grey_scale_frequency[index] / max_grey_scale_frequency ) * 510 )
               
    result_histogram = image_from_matrix( 511, 256, 200 )

    for histogram_position_x in range( 511 ):
        for histogram_position_y in range( 255, 255 - grey_scale_frequency[histogram_position_x // 2], -1 ):
            result_histogram[histogram_position_y][histogram_position_x][R] = 0
            result_histogram[histogram_position_y][histogram_position_x][G] = 0
            result_histogram[histogram_position_y][histogram_position_x][B] = 0
            
    return image_from_matrix( result_histogram )


image1 = load_image_path('/media/zeller/VICTOR/PDI/images/para-somar-A1.jpg')
image2 = load_image_path('/media/zeller/VICTOR/PDI/images/para-somar-A2.jpg')

result_image = multiply_images([ image1, image2], False)
result_image.show("Adicao Imagens")
