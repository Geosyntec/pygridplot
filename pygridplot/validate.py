from matplotlib import pyplot


def figure_axes(ax):
    if ax is None:
        fig, ax = pyplot.subplots()
    elif isinstance(ax, pyplot.Axes):
        fig = ax.figure
    else:
        raise ValueError("`ax` is not an Axes object")

    return fig, ax
