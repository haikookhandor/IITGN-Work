import os

i = 0
path="Data/"
for filename in os.listdir(path):
    if filename.startswith("Horse"):
        my_dest = "horse" + str(i) + ".jpg"
        my_source = path + filename
        my_dest = path + my_dest
        os.rename(my_source, my_dest)
    i += 1
    
i = 0
path="Data/"
for filename in os.listdir(path):
    if filename.startswith("monkey"):
        my_dest = "monkey" + str(i) + ".jpg"
        my_source = path + filename
        my_dest = path + my_dest
        os.rename(my_source, my_dest)
        i += 1