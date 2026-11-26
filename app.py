import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


###
def sidebar_divider(level=1):
    styles = {
        1: """
            border: none;
            border-top: 3px solid #666;
            margin: 1.5rem 0;
        """,
        2: """
            border: none;
            border-top: 1px solid #999;
            margin: 1rem 0;
        """,
        3: """
            border: none;
            border-top: 1px dashed #bbb;
            width: 80%;
            margin: 0.7rem auto;
        """,
    }

    st.sidebar.markdown(
        f'<hr style="{styles[level]}">',
        unsafe_allow_html=True,
    )
    
    
def plt_image(img_2d, zmin=None, zmax=None, colorscale="Viridis"):
    fig = go.Figure(
        data=go.Heatmap(
            z=img_2d,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Counts/pixel")
        )
    )

    fig.update_xaxes(
        constrain="domain"
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain="domain"
    )

    st.plotly_chart(fig)


def make_profiles_binned(img_2d, x_center, y_center, half_width, mode="mean"):
    ny, nx = img_2d.shape
    y_min = max(0, y_center - half_width)
    y_max = min(ny, y_center + half_width + 1)
    x_strip = img_2d[y_min:y_max, :]

    if mode == "mean":
        x_profile = np.mean(x_strip, axis=0)
    elif mode == "sum":
        x_profile = np.sum(x_strip, axis=0)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

    x_min = max(0, x_center - half_width)
    x_max = min(nx, x_center + half_width + 1)
    y_strip = img_2d[:, x_min:x_max]

    if mode == "mean":
        y_profile = np.mean(y_strip, axis=1)
    elif mode == "sum":
        y_profile = np.sum(y_strip, axis=1)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

    x_axis = np.arange(nx)
    y_axis = np.arange(ny)

    return x_axis, x_profile, y_axis, y_profile


def plt_profile(axis_1d, profile_1d, title="", x_label="Pixel", y_label="Counts/pixel", vline=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=axis_1d,
            y=profile_1d,
            mode="lines",
            name=title
        )
    )

    if vline is not None:
        fig.add_vline(
            x=vline,
            line_width=1.5,
            line_dash="dash",
            line_color="red"
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=300,
        margin=dict(l=50, r=30, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
def show_image_with_profiles(
    img_2d,
    title,
    x_center,
    y_center,
    half_width,
    mode="mean",
    zmin=None,
    zmax=None,
    colorscale="Viridis"
):
    st.subheader(title)
    fig = go.Figure(
        data=go.Heatmap(
            z=img_2d,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Counts/pixel")
        )
    )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain="domain"
    )

    fig.add_vline(
        x=x_center,
        line_width=1,
        line_dash="dash",
        line_color="white"
    )
    fig.add_hline(
        y=y_center,
        line_width=1,
        line_dash="dash",
        line_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)
    x_axis, x_profile, y_axis, y_profile = make_profiles_binned(
        img_2d=img_2d,
        x_center=x_center,
        y_center=y_center,
        half_width=half_width,
        mode=mode
    )
    
    col1, col2 = st.columns(2)
    with col1:
        plt_profile(
            axis_1d=x_axis,
            profile_1d=x_profile,
            title=f"X profile (binning in Y, ±{half_width} pix)",
            x_label="X [pixel]",
            y_label="Counts/pixel" if mode == "mean" else "Counts",
            vline=x_center
        )

    with col2:
        plt_profile(
            axis_1d=y_axis,
            profile_1d=y_profile,
            title=f"Y profile (binning in X, ±{half_width} pix)",
            x_label="Y [pixel]",
            y_label="Counts/pixel" if mode == "mean" else "Counts",
            vline=y_center
        )
        
        
        
#############################################################
pixel_num_Xtend = 1028
img_2d = np.empty((pixel_num_Xtend, pixel_num_Xtend))

### Random seed setting ###
sidebar_divider(1)
st.sidebar.header("Random seed setting")

if "seed_bgd" not in st.session_state:
    st.session_state.seed_bgd = 42

if "seed_src" not in st.session_state:
    st.session_state.seed_src = 43

if st.sidebar.button("Regenerate background"):
    st.session_state.seed_bgd += 1

if st.sidebar.button("Regenerate source"):
    st.session_state.seed_src += 1

if "seed_ext" not in st.session_state:
    st.session_state.seed_ext = 44

if st.sidebar.button("Regenerate extended background"):
    st.session_state.seed_ext += 1
    
if st.sidebar.button("Regenerate all"):
    st.session_state.seed_bgd += 1
    st.session_state.seed_src += 1
    st.session_state.seed_ext += 1

st.sidebar.write(f"Background seed: {st.session_state.seed_bgd}")
st.sidebar.write(f"Source seed: {st.session_state.seed_src}")

rng_bgd = np.random.default_rng(st.session_state.seed_bgd)
rng_src = np.random.default_rng(st.session_state.seed_src)
rng_ext = np.random.default_rng(st.session_state.seed_ext)

st.sidebar.write(
    f"Extended background seed: {st.session_state.seed_ext}"
)

######==< streamlit >==######
st.title("Wavelet Simulation")


### sidebar ###
sidebar_divider(1)###################
st.sidebar.header("plt setting")
cs_list = st.sidebar.slider(
    "color scale (counts/pixel)",
    min_value=0.0,
    max_value=10.0,
    value=[0.0, 3.0],
    step=0.1,
    format="%.3f"
)
cs_smooth = st.sidebar.slider(
    "Smoothing color scale (counts/pixel)",
    min_value=0.0,
    max_value=10.0,
    value=[0.0, 1.5],
    step=0.01,
    format="%.3f"
)
sidebar_divider(1)#########################
st.sidebar.header("image setting")
st.sidebar.subheader("Back Ground")
bgd = st.sidebar.number_input(
    "counts (counts/pixel)",
    value=0.03,
    min_value=0.0,
    step=0.01,
    format="%.3f"
)
sidebar_divider(2)
st.sidebar.subheader("Extended background")
enable_ext = st.sidebar.checkbox(
    "Enable extended background",
    value=False
)
x_ext = st.sidebar.number_input(
    "Extended background x (pixel)",
    value=400,
    min_value=0,
    max_value=pixel_num_Xtend-1,
    step=10,
    disabled=not enable_ext
)
y_ext = st.sidebar.number_input(
    "Extended background y (pixel)",
    value=480,
    min_value=0,
    max_value=pixel_num_Xtend-1,
    step=10,
    disabled=not enable_ext
)
n_ext = st.sidebar.number_input(
    "Peak intensity (counts/pixel)",
    value=0.3,
    min_value=0.0,
    step=0.01,
    format="%.3f",
    disabled=not enable_ext
)
sigma_ext = st.sidebar.number_input(
    "Extended background sigma (pixel)",
    value=150,
    min_value=1,
    step=10,
    disabled=not enable_ext
)
sidebar_divider(2)##########################
st.sidebar.subheader("Source setting")
num_sources = st.sidebar.number_input(
    "Number of sources",
    min_value=0,
    value=1,
    step=1
)
sources = []

for i in range(num_sources):
    with st.sidebar.expander(
        f"Source {i+1}",
        expanded=(i == 0)
    ):
        default_pos = (
            pixel_num_Xtend // 2 + i * 60
        ) % pixel_num_Xtend

        x_src = st.number_input(
            "x (pixel)",
            min_value=0,
            max_value=pixel_num_Xtend-1,
            value=default_pos,
            step=1,
            key=f"x_src_{i}"
        )

        y_src = st.number_input(
            "y (pixel)",
            min_value=0,
            max_value=pixel_num_Xtend-1,
            value=default_pos,
            step=1,
            key=f"y_src_{i}"
        )

        n_src = st.number_input(
            "Peak intensity (counts/pixel)",
            min_value=0.0,
            value=0.7,
            step=0.01,
            format="%.3f",
            key=f"n_src_{i}"
        )

        sigma_src = st.number_input(
            "Sigma (pixel)",
            min_value=1,
            value=25,
            step=1,
            key=f"sigma_src_{i}"
        )

        sources.append({
            "x": x_src,
            "y": y_src,
            "counts": n_src,
            "sigma": sigma_src
        })
sigma_source = st.sidebar.number_input(
    "pixel (pixel scale = 1.6''/pixel)",
    value = 25,
    min_value=1,
    step=5
)
sidebar_divider(2)###############################
st.sidebar.subheader("Profile setting")
if num_sources > 0:

    profile_index = st.sidebar.selectbox(
        "Profile target",
        options=list(range(num_sources)),
        format_func=lambda i: f"Source {i+1}"
    )

    x_source = sources[profile_index]["x"]
    y_source = sources[profile_index]["y"]
    sigma_source = sources[profile_index]["sigma"]

else:

    x_source = pixel_num_Xtend // 2
    y_source = pixel_num_Xtend // 2
    sigma_source = 25


sidebar_divider(1)#####################
st.sidebar.header("Wavelet setting")
sigma_wavelet = st.sidebar.number_input(
    "Wavelet scale (pixel)",
    value=4.0,
    min_value=1.4,
    max_value=8.0,
    step=0.1
)
################


x = np.arange(pixel_num_Xtend)
y = np.arange(pixel_num_Xtend)
X, Y = np.meshgrid(x, y)


### poisson noise image ###
st.divider()
st.header(f"Back ground ({bgd} counts/pixel)")
img_bgd = rng_bgd.poisson(lam=bgd, size=img_2d.shape)
plt_image(img_2d=img_bgd, zmin=cs_list[0], zmax=cs_list[1])


### Extended background image ###
if enable_ext:
    r_squared_ext = (
        (X - x_ext)**2 + (Y - y_ext)**2
    )

    ext_model = n_ext * np.exp(
        -r_squared_ext / (2 * sigma_ext**2)
    )

    img_ext = rng_ext.poisson(lam=ext_model)
    
    st.divider()
    show_image_with_profiles(
        img_2d=img_ext,
        title="Extended background",
        x_center=pixel_num_Xtend/2,
        y_center=pixel_num_Xtend/2,
        half_width=pixel_num_Xtend/2,
        mode="mean",
        zmin=cs_list[0],
        zmax=cs_list[1],
        colorscale="Viridis"
    )
else:
    ext_model = np.zeros_like(img_bgd, dtype=float)
    img_ext = np.zeros_like(img_bgd)
    
    
### Source image ###
st.divider()
st.header(f"Source image ({num_sources} sources)")
src_model = np.zeros_like(X, dtype=float)
for src in sources:

    r_squared = (
        (X - src["x"])**2
        + (Y - src["y"])**2
    )

    src_model += src["counts"] * np.exp(
        -r_squared / (2 * src["sigma"]**2)
    )

img_src = rng_src.poisson(lam=src_model)

plt_image(
    img_2d=img_src,
    zmin=cs_list[0],
    zmax=cs_list[1]
)

### (Poisson noise) + (Source) image ###
st.divider()
st.header("Source + Back ground")
img_src_bgd = img_bgd + img_ext + img_src

half_width_profile = int(sigma_source)

show_image_with_profiles(
    img_2d=img_src_bgd,
    title="Source + Background",
    x_center=x_source,
    y_center=y_source,
    half_width=half_width_profile,
    mode="mean",
    zmin=cs_list[0],
    zmax=cs_list[1],
    colorscale="Viridis"
)

### Gaussian smoothing image ###
st.divider()
st.header("Gaussian smoothing image")
img_smooth = gaussian_filter(
    img_src_bgd.astype(float),
    sigma=sigma_wavelet
)

show_image_with_profiles(
    img_2d=img_smooth,
    title="Gaussian smoothing",
    x_center=x_source,
    y_center=y_source,
    half_width=half_width_profile,
    mode="mean",
    zmin=cs_smooth[0],
    zmax=cs_smooth[1],
    colorscale="Viridis"
)

### Laplacian image ###
st.divider()
st.header("Laplacian image (Mexican-hat function)")
img_wavelet = -(sigma_wavelet**2) * laplace(img_smooth)
vmax = np.max(np.abs(img_wavelet))
show_image_with_profiles(
    img_2d=img_wavelet,
    title="Laplacian image (Mexican-hat function)",
    x_center=x_source,
    y_center=y_source,
    half_width=half_width_profile,
    mode="mean",
    zmin=-vmax,
    zmax=vmax,
    colorscale="RdBu_r"
)
