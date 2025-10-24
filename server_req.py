import requests
import json

from car import Car

BASE_URL = "http://sddec25-09e.ece.iastate.edu:8080/api"
LOT_ID = 0

def post_block( block, cars ):
    data = { 
            "cars" : [ car.to_dict() for car in cars ]
        }
    print( data )

    return requests.post( 
        BASE_URL + f"/parkinglots/{ LOT_ID }/{ block }", 
        json = {
            "cars" : [ car.to_dict() for car in cars ]
        })

def get_lots( ):
    return requests.get( BASE_URL + "/parkinglots" ).text

if __name__ == '__main__':
    car1 = Car()
    car2 = Car( color='red', style='truck', plates=[ 'IIJ602', 'GGG777', 'DOG' ] )

    cars = [ car1, car2 ]
    print( post_block( block=2, cars=cars ) )
    # print( get_lots() )
