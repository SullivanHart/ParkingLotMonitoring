import requests
import json

from car import Car
from plate import Plate

BASE_URL = "http://sddec25-09e.ece.iastate.edu:8080/api"
LOT_ID = 0

def post_block( block, cars ):
    json = { 
            "cars" : [ car.to_dict() for car in cars ] if cars else []
        }
    print( f"[INFO] JSON: {json}" )

    return requests.post( 
        BASE_URL + f"/parkinglots/{ LOT_ID }/{ block }", 
        json = json)

def get_lots( ):
    return requests.get( BASE_URL + "/parkinglots" ).text

if __name__ == '__main__':
    car1 = Car()

    plate1 = Plate( text='IIJ123', prob=.47 )
    plate2 = Plate( text='DOG', prob=.68 )
    plate3 = Plate( text='BCH023', prob=.236 )

    car2 = Car( color='red', style='truck', plates=[ plate1, plate2, plate3 ] )

    cars = [ car1, car2 ]
    print( post_block( block=2, cars=cars ) )
    # print( get_lots() )
