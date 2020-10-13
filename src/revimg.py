def reverse(image):
    height, width = img.shape

    #size of each image in mnist dataset
    rev = np.zeros((28,28))

    for h in range (height):
        for w in range (width):
            if (rev[h][w] == 0):
                rev[h][w] = 1 
            else:
                rev[h][w] = 0

    return rev

if __name__ == "__main__":
    print ("Need to call as a module")
else:
    import numpy as np
