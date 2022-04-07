import streamlit as st
# caching option only for reset-button
#from streamlit import caching

import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.font_manager

#import random

# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

#############################################
# Define the function that updates the plot #
#############################################

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(xs,ys,x_int,y_int, integral_sum, integral_sum_real):
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
    dt = round(length/10)
    
    ymin = min(ys)
    ymax = max(ys)
    heigth = ymax-ymin
    dy = round(heigth/10)
    
    handles = st.session_state.handles

    ax = st.session_state.mpl_fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot the data points
        handles["function"] = ax.plot(xs, ys,
                                        color='g',
                                        #linewidth=0,
                                        #marker='o',
                                        #ms=1,
                                        label='function f(x)')[0]#.format(degree))[0]

        # # plot f and append the plot handle
        # handles["integral"] = ax.plot(xs, y_int,
        #                               color='b',
        #                               label="integral")[0]

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Integration of a function')
        plt.xlabel('x', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        for k, spine in ax.spines.items():
            spine.set_zorder(0)

        # set the axes locations and style
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_color('none')
        
        # show integral
        verts = [(tmin, 0), *zip(xs, y_int), (tmax, 0)]
        handles["poly"] = Polygon(verts, facecolor='b', edgecolor='b',label="integral F(x)",fill=True,alpha=0.5)
        handles["filling"] = ax.add_patch(handles["poly"])

    else:
        ###################
        # Update the plot #
        ###################

        # Update the data points plot
        handles["function"].set_xdata(xs)
        handles["function"].set_ydata(ys)

        # # update the integral plot
        # handles["integral"].set_xdata(xs)
        # handles["integral"].set_ydata(y_int)
        
        # remove old integral
        handles["filling"].remove()
        # show new integral
        verts = [(tmin, 0), *zip(xs, y_int), (tmax, 0)]
        handles["poly"].set_xy(verts)
        handles["filling"] = ax.add_patch(handles["poly"])
        
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
        yticks = [x for x in np.arange(round(min(ymin,0)),round(ymax),dy).round(1)]
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
    ax.set_ylim([(min(ymin,0)-0.5), ymax+0.5])
    

    # show legend
    handles["poly"].set_label('integral A = ' + str(integral_sum) + ', actual integral is ' + str(integral_sum_real))
    legend_handles = [handles["function"],handles["poly"]]
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()

def interpret_f(f_input,xs):
    if f_input:
        f = lambda x: eval(f_input)#literal_eval(f_input)
    else:
        f = lambda x: 0
    ys = np.array([f(x) for x in xs])
    return f,ys

def create_data(inttype,n,xs,f,n_int,x_int):
    if inttype == 'left Riemann sum':
        y_int = np.zeros(n)
        i=0
        for x in xs:
            i_eval = np.searchsorted(x_int,x,side='right')-1#*int(n/n_int)
            x_eval = x_int[i_eval]
            y_int[i] = f(x_eval)
            i += 1
    elif inttype == 'right Riemann sum':
        y_int = np.zeros(n)
        i=0
        for x in xs:
            i_eval = np.searchsorted(x_int,x,side='left')#*int(n/n_int)
            x_eval = x_int[i_eval]
            y_int[i] = f(x_eval)
            i += 1
    elif inttype == 'Trapezoidal Riemann sum':
        y_int = np.zeros(n)
        i=0
        for x in xs:
            i_left = np.searchsorted(x_int,x,side='right')-1
            i_right= np.searchsorted(x_int,x,side='left')#*int(n/n_int)
            x1 = x_int[i_left]
            x2 = x_int[i_right]
            y1 = f(x1)
            y2 = f(x2)
            # y = (y2-y1)/(x2-x1) * x + (x2y1-x1y2)/(x2-x1)
            if x1 == x2:
                y_int[i] = y1
            else:
                y_int[i] = (y2-y1)/(x2-x1) * x + (x2*y1-x1*y2)/(x2-x1)
            i += 1
    integral_sum = round(sum(y_int/len(xs)),2)
    integral_sum_real = round(sum(ys/len(xs)),2)
    return y_int, integral_sum, integral_sum_real

def clear_figure():
    del st.session_state['mpl_fig']
    del st.session_state['handles']

###############################################################################
# main
###############################################################################
# create sidebar widgets

st.sidebar.title("Advanced settings")

# Data options
st.sidebar.markdown("Data Options")

n = st.sidebar.number_input(
            'resolution',
            min_value=500,
            max_value=5000,
            value=1000)
    
xmin = st.sidebar.number_input('xmin',
                       min_value = 0,
                       max_value = 50,
                       value = 0)

xmax = st.sidebar.number_input('xmax',
                       min_value = 0,
                       max_value = 50,
                       value = 10)

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
        st.title('Integration of a function f(x)')
    with tcol2:
        st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                    'https://share.streamlit.io/PhiSpel/spielbeispiel-interpolation/main" width="200"/>',
                    unsafe_allow_html=True)
else:
    st.title('Approximated Data Points')
        
col1,col2,col3 = st.columns(3)
with col1:
    f_input = st.text_input(label='input your function',
                         value='0.2*x**2 + 0.5 - x*math.sin(x)',
                         help='''type e.g. 'math.sin(x)' to generate a sine function''')

with col2:
    inttype = st.selectbox(label='integration type',
                           options=('left Riemann sum','right Riemann sum','Trapezoidal Riemann sum'),
                           index=1)

with col3:
    n_int = st.number_input(label='number of points to integrate between',
                            min_value = 1,
                            max_value = 300,
                            value = 8)
    
xs = np.linspace(xmin,xmax,n)
x_int = np.linspace(xmin,xmax,n_int)
f,ys = interpret_f(f_input,xs)
y_int, integral_sum, integral_sum_real = create_data(inttype,n,xs,f,n_int,x_int)
    
##############################################################################
# Plotting

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
update_plot(xs,ys,x_int,y_int, integral_sum, integral_sum_real)
st.pyplot(st.session_state.mpl_fig)