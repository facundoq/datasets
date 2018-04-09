import numpy as np

def draw_square_inner(image,position,radius,color,fill=False):
    topleft      = position - radius
    bottomright  = position + radius+1
    topleft[0]=max(0,topleft[0])
    topleft[1]=max(0,topleft[1])
    bottomright[0]=min(image.shape[0],bottomright[0])
    bottomright[1]=min(image.shape[1],bottomright[1])

    if fill:
        bottomright=bottomright-1
        image[topleft[0]:bottomright[0],topleft[1]:bottomright[1],]=color
    else:
        image[topleft[0],topleft[1]:bottomright[1],]=color
        image[bottomright[0]-1,topleft[1]:bottomright[1],]=color
#         print(bottomright[0]-1)
#         print(topleft[1],bottomright[1])
        image[topleft[0]:bottomright[0],topleft[1],]=color
        image[topleft[0]:bottomright[0],bottomright[1]-1,]=color
def draw_square(image,position,radius,color,fill=False,thickness=1):
    if fill:
        draw_square_inner(image,position,radius,color,fill)
    else:
        for i in range(thickness):
            draw_square_inner(image,position,radius+i,color,fill)

def draw_positions(image,positions):
    color=255
    radiuses={'head':35,'left_hand':30,'right_hand':30}
    for k,position in positions.items():
        draw_square(image,position,radiuses[k],color,thickness=3)