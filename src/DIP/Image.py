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
HISTOGRAM_HEIGHT = 256
HISTOGRAM_WIDTH = 511
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


def new_laplacian_kernel ( lines = 3, columns = 3, pound = -8 ):
    """Cria kernel laplacianao 
    Args:
        lines: número de linhas do kernel, default = 3
        columns: número de colunas do kernel, default = 3
        pound: peso central do kernel. Default = -8
    Returns:
        kernel (lines x columns)
    """
    laplacian_kernel = [ [ line * 0 + 1 for line in range( lines ) ] for column in range( columns ) ]
    
    laplacian_kernel[start_kernel_height( laplacian_kernel )][start_kernel_width( laplacian_kernel )] = pound
    
    return laplacian_kernel


def find_kernel_width( kernel ):
    """Descobre largura do kernel (width)
    Args:
        kernel: kernel que se deseja-se descobrir largura
    Returns:
        largura do kernel
    """
    return len( kernel[0] )


def find_kernel_height( kernel ):
    """Descobre altura do kernel (width)
    Args:
        kernel: kernel que se deseja-se descobrir altura
    Returns:
        altura do kernel
    """
    return len( kernel )


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
    return len( image[0] )


def get_image_height ( image ):
    return len( image )


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
        for position_x in range( matrix_image_height ):
            for position_y in range( matrix_image_width ):
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
    
    for position_x in range( image_height ):
        for position_y in range( image_width ):
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
        for position_x in range( new_image_height ):
            for position_y in range( new_image_width ):
                base_matrix_image[position_x][position_y] = ( 
                                   image1[position_y, position_x][R] + image2[position_y, position_x][R],
                                   image1[position_y, position_x][G] + image2[position_y, position_x][G],
                                   image1[position_y, position_x][B] + image2[position_y, position_x][B]
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
        for position_x in range( new_image_height ):
            for position_y in range( new_image_width ):
                baseImage[position_x][position_y] = ( 
                                   image1[position_y, position_x][R] - image2[position_y, position_x][R],
                                   image1[position_y, position_x][G] - image2[position_y, position_x][G],
                                   image1[position_y, position_x][B] - image2[position_y, position_x][B]
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
        for position_x in range( new_image_height ):
            for position_y in range( new_image_width ):
                baseImage[position_x][position_y] = ( 
                                   image1[position_y, position_x][R] * image2[position_y, position_x][R],
                                   image1[position_y, position_x][G] * image2[position_y, position_x][G],
                                   image1[position_y, position_x][B] * image2[position_y, position_x][B]
                                  )
    if( normalize_result ):
        baseImage = normalize_rgb_image( baseImage )
        
    return image_from_matrix( baseImage )


def half_median_filter ( image, kernel = new_kernel( 3, 3 ) ):
    
    result_image = matrix_from_image( image.height, image.width )
    base_image = load_image_data( image )
    
    for image_position_x in range( image.height ):
        for image_position_y in range( image.width ):
            
            start_position_x = 0 if image_position_x - start_kernel_height( kernel ) <= 0 else image_position_x - start_kernel_height( kernel )
            end_position_x = image_position_x if image_position_x + start_kernel_height( kernel ) >= image.height - 1 else image_position_x + start_kernel_height( kernel )
            
            start_position_y = 0 if image_position_y - start_kernel_width( kernel ) <= 0 else image_position_y - start_kernel_width( kernel )
            end_position_y = image_position_y if image_position_y + start_kernel_width( kernel ) >= image.width - 1 else image_position_y + start_kernel_width( kernel )
            
            kernelSize = ( end_position_x + 1 - start_position_x ) * ( end_position_y + 1 - start_position_y )
            
            for band in range( BANDS ):
                for kernel_position_x in range( start_position_x, end_position_x + 1 ):
                    kernelX = ( end_position_x ) - kernel_position_x
                    for kernel_position_y in range( start_position_y, end_position_y + 1 ):
                        kernelY = ( end_position_y ) - kernel_position_y
                        result_image[image_position_x][image_position_y][band] += base_image[start_position_y + kernelY, start_position_x + kernelX][band] * kernel[kernelY][kernelX]
                        
                result_image[image_position_x][image_position_y][band] /= kernelSize
            
    return image_from_matrix( result_image )


def median_filter ( image, kernel = new_kernel( 3, 3 ) ):
    
    result_image = matrix_from_image( image.height, image.width )
    base_image = load_image_data( image )
    
    start_kernel_x = start_kernel_height( kernel )
    start_kernel_y = start_kernel_width( kernel )
    
    neighborhood_pixel_values = []
    
    for image_position_x in range( image.height ):
        
        start_position_x = 0 if image_position_x - start_kernel_x <= 0 else image_position_x - start_kernel_x
        end_position_x = image_position_x if image_position_x + start_kernel_x >= image.height - 1 else image_position_x + start_kernel_x
        
        for image_position_y in range( image.width ):
            
            start_position_y = 0 if image_position_y - start_kernel_y <= 0 else image_position_y - start_kernel_y
            end_position_y = image_position_y if image_position_y + start_kernel_y >= image.width - 1 else image_position_y + start_kernel_y
            
            for band in range( BANDS ):
                neighborhood_pixel_values = []
                for kernel_position_x in range( start_position_x, end_position_x + 1 ):
                    kernelX = ( end_position_x ) - kernel_position_x
                    for kernel_position_y in range( start_position_y, end_position_y + 1 ):
                        kernelY = ( end_position_y ) - kernel_position_y
                        neighborhood_pixel_values.append( base_image[start_position_y + kernelY, start_position_x + kernelX ][band] )
                        
                result_image[image_position_x][image_position_y][band] = np.median( neighborhood_pixel_values )
            
    return image_from_matrix( result_image )


def laplacian_filter( image, kernel = new_laplacian_kernel( 3, 3 ) ):
    result_image = matrix_from_image( image.height, image.width, pixel_value = 0 )
    base_image = load_image_data( image )
    
    start_kernel_x = start_kernel_height( kernel )
    start_kernel_y = start_kernel_width( kernel )
    
    for image_position_x in range( image.height ):
        
        start_position_x = 0 if image_position_x - start_kernel_x <= 0 else image_position_x - start_kernel_x
        end_position_x = image_position_x if image_position_x + start_kernel_x >= image.height - 1 else image_position_x + start_kernel_x
        
        for image_position_y in range( image.width ):
            
            start_position_y = 0 if image_position_y - start_kernel_y <= 0 else image_position_y - start_kernel_y
            end_position_y = image_position_y if image_position_y + start_kernel_y >= image.width - 1 else image_position_y + start_kernel_y
            
            for band in range( BANDS ):
                final_pixel_value = 0
                for kernel_position_x in range( start_position_x, end_position_x + 1 ):
                    kernelX = ( end_position_x ) - kernel_position_x
                    for kernel_position_y in range( start_position_y, end_position_y + 1 ):
                        kernelY = ( end_position_y ) - kernel_position_y
                        final_pixel_value = final_pixel_value + base_image[start_position_y + kernelY, start_position_x + kernelX][band] * kernel[kernelY][kernelX]
                
                result_image[image_position_x][image_position_y][band] = 0 if final_pixel_value <= 0 else final_pixel_value
    
    return image_from_matrix( normalize_rgb_image( result_image ) )


def brightness_monocromatization ( image ):
    base_image = load_image_data( image )
    base_matrix_image = matrix_from_image( image.height, image.width )
    
    for image_position_x in range( image.height ):
        for image_position_y in range ( image.width):
            chanelValues = [
                base_image[image_position_y, image_position_x][R],
                base_image[image_position_y, image_position_x][G],
                base_image[image_position_y, image_position_x][B]
            ]
            
            base_matrix_image[image_position_x][image_position_y][R] = ( max( chanelValues ) + min( chanelValues ) ) / 2
            base_matrix_image[image_position_x][image_position_y][G] = ( max( chanelValues ) + min( chanelValues ) ) / 2
            base_matrix_image[image_position_x][image_position_y][B] = ( max( chanelValues ) + min( chanelValues ) ) / 2
    
    return image_from_matrix( base_matrix_image )


def median_monocromatization( image ):
    base_image = load_image_data( image )
    base_matrix_image = matrix_from_image( image.height, image.width )
    
    for image_position_x in range( image.height ):
        for image_position_y in range ( image.width):
            chanel_values = base_image[image_position_y, image_position_x][R] + base_image[image_position_y, image_position_x][G] + base_image[image_position_y, image_position_x][B]
                            
            base_matrix_image[image_position_x][image_position_y][R] = chanel_values / 3
            base_matrix_image[image_position_x][image_position_y][G] = chanel_values / 3
            base_matrix_image[image_position_x][image_position_y][B] = chanel_values / 3
    
    return image_from_matrix( base_matrix_image )


def luminosity_monocromatization( image, luminosity_mode = "BT709" ):
    chanel_pound = PUND_LUMINOSITY_MODES.get( luminosity_mode )
    base_image = load_image_data( image )
    base_matrix_image = matrix_from_image( image.height, image.width )
    
    for image_position_x in range( image.height ):
        for image_position_y in range ( image.width ):
            chanel_values = [base_image[image_position_y, image_position_x][R], base_image[image_position_y, image_position_x][G], base_image[image_position_y, image_position_x][B]]
            base_matrix_image[image_position_x][image_position_y][R] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
            base_matrix_image[image_position_x][image_position_y][G] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
            base_matrix_image[image_position_x][image_position_y][B] = ( chanel_values[R] * chanel_pound[R] ) + ( chanel_values[G] * chanel_pound[G] ) + ( chanel_values[B] * chanel_pound[B] )
    
    return image_from_matrix( base_matrix_image )


def generate_relative_histogram( image ):
    grey_scale_frequency = generate_grey_scale_frequence(image)
    max_grey_scale_frequency = get_max_grey_scale_frequency( grey_scale_frequency )
    
    adjustment_factor = HISTOGRAM_HEIGHT / max_grey_scale_frequency 

    for index in range( len( grey_scale_frequency ) ):
        grey_scale_frequency[index] = int (adjustment_factor * grey_scale_frequency[index] ) 
               
    result_histogram = matrix_from_image( HISTOGRAM_HEIGHT, HISTOGRAM_WIDTH, pixel_value = 200 )

    for histogram_position_y in range( HISTOGRAM_WIDTH ):
        for histogram_position_x in range( HISTOGRAM_HEIGHT - grey_scale_frequency[histogram_position_y // 2] -1, HISTOGRAM_HEIGHT ):
            result_histogram[histogram_position_x][histogram_position_y][R] = 0
            result_histogram[histogram_position_x][histogram_position_y][G] = 0
            result_histogram[histogram_position_x][histogram_position_y][B] = 0
            
    return image_from_matrix( result_histogram )


def generate_absolute_histogram(image):
    pass

def generate_grey_scale_frequence(image):
    grey_scale_frequency = [i * 0 for i in range( 256 )]
    base_image = load_image_data(image)
    
    for image_position_x in range( image.height ):
        for image_position_y in range( image.width):
            pixel_value = base_image[image_position_y, image_position_x][R]
            grey_scale_frequency[pixel_value] += 1
    
    return grey_scale_frequency
    
def get_max_grey_scale_frequency(grey_scale_frequence):
    return max( grey_scale_frequence )
#===============================================================================
#                                  TESTES
#===============================================================================
# image = load_image_path( '/media/zeller/VICTOR/PDI/images/salt_pepper_vih.png' )
# image.show()
# image = median_filter( image , new_kernel(5, 5))
# image.show()
# laplacian_image = laplacian_filter( image )
# laplacian_image.show()

#===============================================================================
#                          TESTES MONOCORMATIZACAO
#===============================================================================
mono = load_image_path( '/media/zeller/VICTOR/PDI/images/vih.jpeg' )
mono.show()
#brightness_monocromatization(mono).show()
#median_monocromatization(mono).show()
mono = luminosity_monocromatization(mono)
mono.show()
generate_relative_histogram(mono).show()

