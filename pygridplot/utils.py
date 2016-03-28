import pandas


def addSecondColumnLevel(levelval, levelname, olddf):
    '''
    Takes a simple index on a dataframe's columns and adds a new level
    with a single value.
    E.g., df.columns = ['res', 'qual'] -> [('Infl' ,'res'), ('Infl', 'qual')]
    '''
    if isinstance(olddf.columns, pandas.MultiIndex):
        raise ValueError('Dataframe already has MultiIndex on columns')
    colarray = [[levelval]*len(olddf.columns), olddf.columns]
    colindex = pandas.MultiIndex.from_arrays(colarray)
    newdf = olddf.copy()
    newdf.columns = colindex
    newdf.columns.names = [levelname, 'quantity']
    return newdf
