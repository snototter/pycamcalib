from matplotlib import pyplot as plt
import math
def plot(g, name):
    plt.axis([-10, 10, -10, 10])
    ax = plt.gca()
    ax.yaxis.grid(color='gray')
    ax.xaxis.grid(color='gray')

    X, Y = [], []
    for i in range(100):
        (r, (x, y)) = next(g)
        X.append(x)
        Y.append(y)
        print ("{:d}: radius {}".format(i, r))

    plt.plot(X, Y, 'r-', linewidth=2.0)
    plt.title(name)
    plt.savefig(name + ".png")

def helicalIndices(n):
    num = 0
    curr_x, dir_x, lim_x, curr_num_lim_x = 0, 1, 1, 2
    curr_y, dir_y, lim_y, curr_num_lim_y = -1, 1, 1, 3
    curr_rep_at_lim_x, up_x = 0, 1
    curr_rep_at_lim_y, up_y = 0, 1

    while num < n:
        if curr_x != lim_x:
            curr_x +=  dir_x
        else:
            curr_rep_at_lim_x += 1
            if curr_rep_at_lim_x == curr_num_lim_x - 1:
                if lim_x < 0:
                    lim_x = (-lim_x) + 1
                else:
                    lim_x = -lim_x
                curr_rep_at_lim_x = 0
                curr_num_lim_x += 1
                dir_x = -dir_x
        if curr_y != lim_y:
            curr_y = curr_y + dir_y
        else:
            curr_rep_at_lim_y += 1
            if curr_rep_at_lim_y == curr_num_lim_y - 1:
                if lim_y < 0:
                    lim_y = (-lim_y) + 1
                else:
                    lim_y = -lim_y
                curr_rep_at_lim_y = 0
                curr_num_lim_y += 1
                dir_y = -dir_y
        r = math.sqrt(curr_x*curr_x + curr_y*curr_y)        
        yield (r, (curr_x, curr_y))
        num += 1

hi = helicalIndices(200)
plot(hi, "helicalIndices")

