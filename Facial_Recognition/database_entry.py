import pandas as pd
from img_to_encoding import img_to_encoding

def database_entry(FRmodel):
    database = {}
    database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
    database["hamza"] = img_to_encoding("images/hamza.jpg", FRmodel)
    
    return database