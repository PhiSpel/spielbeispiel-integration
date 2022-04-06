import streamlit as st
# caching option only for reset-button
#from streamlit import caching

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.font_manager

import random

# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

#############################################
# Define the function that updates the plot #
#############################################

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot():
    xs, data, approx, f_input, show_solution, ticks_on = st.session_state.xs, st.session_state.data, st.session_state.approx, st.session_state.f_input, st.session_state.show_solution, st.session_state.ticks_on
    show_polyfit_solution = st.session_state.show_polyfit_solution
    function = st.session_state.function(xs)
    # Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    # updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    # The figure is stored in st.session_state.fig.

    # :param t0: Evaluation point of the function/Taylor polynomial
    # :param ft0: Function evaluated at t0
    # :param xs: numpy-array of x-coordinates
    # :param ys: numpy-array of f(x)-coordinates
    # :param ps: numpy-array of P(x)-coordinates, where P is the Taylor polynomial
    # :param visible: A flag wether the Taylor polynomial is visible or not
    # :param xmin: minimum x-range value
    # :param xmax: maximum x-range value
    # :param ymin: minimum y-range value
    # :param ymax: maximum y-range value
    # :return: none.
    
    tmin = min(xs)
    tmax = max(xs)
    length = tmax-tmin
    dt = round(length/10,1)
    
    ymin = min(data)
    ymax = max(data)
    heigth = ymax-ymin
    dy = round(heigth/10,1)
    
    if f_input:
        f = lambda x: eval(f_input)
    else:
        f = lambda x: 0
    ys= [f(x) for x in xs]
    
    approx = approx(xs)
        
    handles = st.session_state.handles

    ax = st.session_state.mpl_fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot the data points
        handles["datapoints"] = ax.plot(xs, data,
                                        color='g',
                                        linewidth=0,
                                        marker='o',
                                        ms=1,
                                        label='data points')[0]#.format(degree))[0]

        # plot f and append the plot handle
        handles["f_input"] = ax.plot(xs, ys,
                                      color='b',
                                      label="your best guess")[0]

        # plot actual function and append the plot handle
        handles["actual"] = ax.plot(xs, function,
                                      color='g',
                                      label="actual function")[0]

        handles["actual"].set_visible(show_solution)
        
        # plot approximation and append the plot handle
        handles["approx"] = ax.plot(xs, approx,
                                      color='orange',
                                      label="np.polyfit guess")[0]

        handles["approx"].set_visible(show_polyfit_solution)

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Approximation of a series of data points')
        plt.xlabel('t', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        for k, spine in ax.spines.items():
            spine.set_zorder(0)

        # set the axes locations and style
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_color('none')

    else:
        ###################
        # Update the plot #
        ###################

        # Update the data points plot
        handles["datapoints"].set_xdata(xs)
        handles["datapoints"].set_ydata(data)

        # update the input plot
        handles["f_input"].set_xdata(xs)
        handles["f_input"].set_ydata(ys)

        # update the input plot
        handles["actual"].set_xdata(xs)
        handles["actual"].set_ydata(function)

        # update the visibility of the Taylor expansion
        handles["actual"].set_visible(show_solution)
        
        # update the input plot
        handles["approx"].set_xdata(xs)
        handles["approx"].set_ydata(approx)

        # update the visibility of the Taylor expansion
        handles["approx"].set_visible(show_polyfit_solution)

    # set x and y ticks, labels and limits respectively
    if ticks_on:
        xticks = [x for x in np.arange(tmin,tmax,dt).round(1)]
    else:
        xticks=[]
    xticklabels = [str(x) for x in xticks]
    
    if tmin <= 0 <= tmax:
        xticks.append(0)
        xticklabels.append("0")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if ticks_on:
        yticks = [x for x in np.arange(round(ymin),round(ymax),dy).round(1)]
    else:
        yticks=[]
    yticklabels = [str(x) for x in yticks]
    if ymin <= 0 <= ymax:
        yticks.append(0)
        yticklabels.append("0")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # set the x and y limits
    ax.set_xlim([tmin-0.5, tmax+0.5])
    ax.set_ylim([ymin-0.5, ymax+0.5])
    

    # show legend
    legend_handles = [handles["datapoints"], ]
    legend_handles.append(handles['f_input'])
    if show_solution:
        legend_handles.append(handles["actual"])
    if show_polyfit_solution:
        legend_handles.append(handles["approx"])
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()

@st.experimental_singleton
def create_new_factors(datatype,f_data_input):
    a = round(random.random(),3)
    b = round(random.random(),3)
    c = round(random.random(),3)
    if datatype == 'constant':
        function = lambda x: a
        factors=[a]
    elif datatype == 'linear':
        #data = ax + b
        function = lambda x: a*x + b
        factors=[a,b]
    elif datatype == 'quadratic':
        #data = ax^2+bx+c
        function = lambda x: a*x**2 + b*x + c
        factors=[a,b,c]
    elif datatype == 'custom':
        if not f_data_input == '':
            function = lambda x: eval(f_data_input)
        else:
            function = lambda x: 0
        factors = ''
    return function, factors

# can only cache one function output, otherwise things get stuck
#@st.experimental_singleton
def create_randomization(dist_type):
    n = st.session_state.n
    length = st.session_state.length
    xs = np.random.rand(n)*length
    xs.sort()
    if dist_type == 'normal':
        dev = np.random.normal(0,st.session_state.sigma,n)
    elif dist_type == 'equal':
        dev = np.random.rand(n)*st.session_state.sigma
    return dev,xs
    
def create_new_points():
    dev,xs = create_randomization(st.session_state.dist_type)
    st.session_state.xs = xs
    st.session_state.data = st.session_state.function(xs) + dev
    return

def reset_rnd():
    st.experimental_singleton.clear()
    request_new_data()
    return

def update_approx():
    xs,data,approxtype = st.session_state.xs,st.session_state.data,st.session_state.approxtype
    if approxtype == 'constant':
        z=np.polyfit(xs,data,0)
    elif approxtype == 'linear':
        z=np.polyfit(xs,data,1)
    elif approxtype == 'quadratic':
        z=np.polyfit(xs,data,2)
    elif approxtype == 'cubic':
        z=np.polyfit(xs,data,3)
    st.session_state.approx=np.poly1d(z)
    return

def request_new_data():
    st.session_state.create_new_data = 1
    return
def request_new_points():
    st.session_state.create_new_points = 1
    return

def clear_figure():
    del st.session_state['mpl_fig']
    del st.session_state['handles']

def write_solution_description():
    solution_description = r'''np.polyfit guesses: $f(x)\approx '''
    factors=np.round(st.session_state.approx,2)
    deg = len(factors)-1
    for i in range(0,deg+1):
        if (factors[i] > 0) & (i > 0):
            solution_description+='+'
        if deg-i > 1:
            solution_description+=str(factors[i]) + 'x^' + str(deg-i)
        elif deg-i == 1:
            solution_description+=str(factors[i]) + 'x'
        elif deg-i == 0:
            solution_description+=str(factors[i])
    solution_description+='''$'''
    st.session_state.solution_description=solution_description
    return

def write_actual_function():
    actual_function = r'''actual function is $f(x) = '''
    if st.session_state.factors == '':
        actual_function+=st.session_state.f_data_input
        actual_function+='''$'''
    else:
        actual_factors = st.session_state.factors
        deg = len(actual_factors)-1
        for i in range(0,deg+1):
            if (actual_factors[i] > 0) & (i > 0):
                actual_function+='+'
            if deg-i > 1:
                actual_function+=str(actual_factors[i]) + 'x^' + str(deg-i)
            elif deg-i == 1:
                actual_function+=str(actual_factors[i]) + 'x'
            elif deg-i == 0:
                actual_function+=str(actual_factors[i])
        actual_function+='''$'''
    st.session_state.actual_function=actual_function
    return

###############################################################################
# setup session_state variables
if 'create_new_data' not in st.session_state:
    st.session_state.create_new_data = 1
if 'create_new_points' not in st.session_state:
    st.session_state.create_new_points = 0
if 'approx' not in st.session_state:
    st.session_state.approx = {}
if 'data' not in st.session_state:
    st.session_state.data = []

###############################################################################
# main
###############################################################################
# create sidebar widgets

st.sidebar.title("Advanced settings")

# Data options

st.session_state.datatype = st.sidebar.selectbox(label="data type",
                            options=('constant','linear','quadratic','custom'),
                            index=2,
                            on_change=request_new_data())

if st.session_state.datatype == 'custom':
    f_data_input = st.sidebar.text_input(label='input the data function',
                         value='0.5*x**2 + 1*x - 2',
                         key='f_data_input',
                         on_change=request_new_data(),
                         help='''type e.g. 'math.sin(x)' to generate a sine function''')
else:
    st.session_state.f_data_input = ''
    
st.session_state.dist_type = st.sidebar.select_slider('select distribution type',
                                         ['normal','equal'],
                                         on_change=request_new_points())

# deviation could be dependent on the total height of the function - but this would need to be re-looped. can't be bothered...
# if 'data' in st.session_state:
#     height = max(st.session_state.data)
# else:
#     height = 100
st.session_state.sigma = st.sidebar.number_input('deviation (standard deviation sigma for normal distribution, range for equal distribution)',
                            min_value=float(0),
                            max_value=float(100),
                            # value=0.1*height,
                            value=float(10),
                            step=0.01,
                            on_change=request_new_points())

st.session_state.n = st.sidebar.slider(
            'number of data points',
            min_value=0,
            max_value=1000,
            value=100,
            on_change=request_new_points())
    

st.session_state.length = st.sidebar.slider('length of interval',
                       min_value = float(1),
                       max_value = float(50),
                       value = float(10),
                       on_change=request_new_points())

# Visualization Options
st.sidebar.markdown("Visualization Options")

# Good for in-classroom use
qr = st.sidebar.checkbox(label="Display QR Code", value=False)

xkcd = st.sidebar.checkbox("use xkcd-style",
                           value=False,
                           on_change=clear_figure)

ticks_on = st.sidebar.checkbox("show xticks and yticks",
                               value=True,
                               on_change=clear_figure,
                               key='ticks_on')

# for now, I will assume matplotlib always works and we dont need the Altair backend
#backend = 'Matplotlib' #st.sidebar.selectbox(label="Backend", options=('Matplotlib', 'Altair'), index=0)

###############################################################################
# Create main page widgets

if qr:
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.title('Approximated Data Points')
    with tcol2:
        st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                    'https://share.streamlit.io/PhiSpel/spielbeispiel-interpolation/main" width="200"/>',
                    unsafe_allow_html=True)
else:
    st.title('Approximated Data Points')
        
st.markdown('''If you want to create a new function or input your own function, change the advanced settings (top-left).''')
st.markdown('''The data points will change at random when you guess a new function, but the function you should guess remains the same!''')

col1,col2 = st.columns(2)
with col1:
    st.session_state.f_input = st.text_input(label='input your guessed function',
                         value='0.2*x**2 + 1*x - 2',
                         help='''type e.g. 'math.sin(x)' to generate a sine function''')

col1,col2,col3,col4 = st.columns(4)
with col1:
    st.session_state.show_solution = st.checkbox("show the actual function",
                                value=False,
                                on_change=clear_figure)
    st.session_state.show_polyfit_solution = st.checkbox("show the np.polyfit solution",
                                value=False,
                                on_change=clear_figure)
if st.session_state.show_polyfit_solution:
    with col2:
        st.session_state.approxtype = st.selectbox(label='approximation type',
                                      options=('constant','linear','quadratic','cubic'),
                                      index=2)
else: st.session_state.approxtype = 'constant'

###
# for some reason, reset still doesn't work...
# with col4:
#     st.button(label='create new randomization',on_click=reset_rnd())
    
if (st.session_state.create_new_data==1):
    st.session_state.function, st.session_state.factors = create_new_factors(st.session_state.datatype,st.session_state.f_data_input)
    create_new_points()
    update_approx()
    st.session_state.create_new_data = 0
    st.session_state.create_new_points = 0
elif (st.session_state.create_new_points==1):
    create_new_points()
    update_approx()
    st.session_state.create_new_points = 0


col1,col2,col3,col4=st.columns(4)
if st.session_state.show_solution:
    write_actual_function()
    with col1:
        st.markdown(st.session_state.actual_function)
if st.session_state.show_polyfit_solution:
    write_solution_description()
    with col2:
        st.markdown(st.session_state.solution_description)

if xkcd:
    # set rc parameters to xkcd style
    plt.xkcd()
else:
    # reset rc parameters to default
    plt.rcdefaults()

# initialize the Matplotlib figure and initialize an empty dict of plot handles
if 'mpl_fig' not in st.session_state:
    st.session_state.mpl_fig = plt.figure(figsize=(8, 3))
    st.session_state.mpl_fig.add_axes([0., 0., 1., 1.])

if 'handles' not in st.session_state:
    st.session_state.handles = {}

# update plot
update_plot()
st.pyplot(st.session_state.mpl_fig)