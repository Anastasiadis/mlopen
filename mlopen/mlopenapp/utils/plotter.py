def plotlify_pie(xy=None, description=""):
    """
    Update the plot data in plotly format.

    :param xy: x and y in a single structure.
    :param description: The description of the plotly plot.
    :param plot_type: The type of the plotly plot.
    :return: A dictionary with the data in plotly format.
    """

    ret = {
        'data': [],
        'layout': {
            'hiddenlabels': ['UNSPECIFIED'],
            'paper_bgcolor': 'rgba(243, 243, 243, 1)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'title': {
                'text': description,
            }
        }
    }

    ret['data'].append(
        {
            'values': [v for k, v in xy.items()],
            'labels': [str(k) for k, v in xy.items()],
            'type': 'pie',
        }
    )
    return ret
