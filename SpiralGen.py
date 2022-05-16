def get_spiral(
        start_x: int = 0,
        start_y: int = 0,
        start_step: int = 0,
        step_end: int = 50,
        step_count: int = 1
    ) -> list:
    x = start_x
    y = start_y
    step = start_step
    sign = +1
    counter = 0
    turn = 'x'

    spiral_coords = []

    while (step < step_end):
        if counter % 2 == 0:
            sign *= -1
            step += step_count
        
        if turn == 'x':
            prev_x = x
            x += step * sign
            for m in range(prev_x, x, (-1 if prev_x > x else 1)):
                spiral_coords.append((m, y))
            turn = 'y'
        else:
            prev_y = y
            y += step * sign
            for n in range(prev_y, y, (-1 if prev_y > y else 1)):
                spiral_coords.append((x, n))
            turn = 'x'
        counter += 1
            

    return spiral_coords

def main():
    from time import perf_counter
    import matplotlib.pyplot as plt

    start = perf_counter()
    spiral = get_spiral()
    end = perf_counter()
    print(f"Time: {end-start}")
    
    for coord in spiral:
        print(coord)

    xs = [ coord[0] for coord in spiral ]
    ys = [ coord[1] for coord in spiral ]
    plt.plot(xs,  ys);
    plt.show();

if ( __name__ == "__main__" ):
    main()