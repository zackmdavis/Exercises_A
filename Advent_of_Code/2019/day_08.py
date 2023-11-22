import itertools
from collections import Counter

with open('input') as f:
    in_content = f.read().rstrip()
    data = in_content

def the_first_star():
    image_dims = 25 * 6
    layer_count = len(data)//image_dims

    layers = []
    fewest_zeroes_index = None
    fewest_zeroes = float('inf')
    for layer_index in range(layer_count):
        layer = data[image_dims*layer_index: image_dims*(layer_index+1)]
        layers.append(layer)
        zero_count = len([p for p in layer if p == '0'])
        if zero_count < fewest_zeroes:
            fewest_zeroes = zero_count
            fewest_zeroes_index = layer_index

    fewest_zeroes_layer = layers[fewest_zeroes_index]
    one_count = len([p for p in fewest_zeroes_layer if p == '1'])
    two_count = len([p for p in fewest_zeroes_layer if p == '2'])
    return one_count * two_count


# This one was very cute!!
def the_second_star():
    grid = []
    for i in range(6):
        row = []
        for j in range(25):
            for k in range(100):
                pixel = data[i*25 + j + k*150]
                if pixel == '0':
                    row.append('■')
                    break
                elif pixel == '1':
                    row.append('_')
                    break
        grid.append(row)
    for row in grid:
        print(''.join(row))


if __name__ == "__main__":
    print(the_first_star())
    print(the_second_star())
