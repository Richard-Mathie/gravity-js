levels[0].interpolate(
    levels[1].interpolate(
        levels[2].interpolate(
        levels[2].decimate(
    levels[1].decimate(
levels[0].decimate(levels[0].rho))))))


render(levels[0].interpolate(
        levels[1].interpolate(
        levels[2].interpolate(
        levels[3].interpolate(
        levels[4].interpolate(
        levels[5].interpolate(
        levels[6].interpolate(
        [[0,0,0.9,0.9],
         [0,0,0.9,0.9],
         [0.1,0.9,0,0],
         [0.3,0.5,0,0]]
        ))))))))

