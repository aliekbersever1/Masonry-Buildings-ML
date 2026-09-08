import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Load metadata and trained models
# ============================================================

@st.cache_resource
def load_all():
    with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models = {}
    for target, target_models in metadata["models"].items():
        models[target] = {}

        for name, relative_path in target_models.items():
            models[target][name] = joblib.load(
                BASE_DIR / relative_path
            )

    return metadata, models


# ============================================================
# Explicit regression equations
# ============================================================

def linear_formula_f1(B, L, H, E, rho):
    return (
        -1.234
        + 0.233 * B
        + 0.531 * L
        - 0.738 * H
        + 3.144 * math.sqrt(E / rho)
    )


def symbolic_formula_f1(B, L, H, E, rho):
    return (
        math.sqrt(E / rho)
        * (1.273 * L / H - 0.395)
        / ((L / (B + 5.428)) ** 4 + 0.278)
    )


# ============================================================
# Helper functions
# ============================================================

def get_valid_ranges(ranges, roles):
    """
    Return the parameter ranges corresponding to Table 1.
    """

    return {
        "B": (
            float(ranges[roles["B"]]["min"]),
            float(ranges[roles["B"]]["max"]),
        ),
        "L": (
            float(ranges[roles["L"]]["min"]),
            float(ranges[roles["L"]]["max"]),
        ),
        "H": (
            float(ranges[roles["H"]]["min"]),
            float(ranges[roles["H"]]["max"]),
        ),
        "E": (
            float(ranges[roles["E"]]["min"]),
            float(ranges[roles["E"]]["max"]),
        ),
        "rho": (
            float(ranges[roles["rho"]]["min"]),
            float(ranges[roles["rho"]]["max"]),
        ),
    }


def check_extrapolation(B, L, H, E, rho, valid_ranges):
    """
    Check whether any user input falls outside the numerical
    parameter domain used for model development.
    """

    input_values = {
        "B": B,
        "L": L,
        "H": H,
        "E": E,
        "rho": rho,
    }

    outside = {}

    for parameter, value in input_values.items():
        minimum, maximum = valid_ranges[parameter]

        if value < minimum or value > maximum:
            outside[parameter] = {
                "value": value,
                "min": minimum,
                "max": maximum,
            }

    return outside


# ============================================================
# Load files
# ============================================================

metadata, models = load_all()

roles = metadata["roles"]
ranges = metadata["ranges"]

valid_ranges = get_valid_ranges(ranges, roles)


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="Masonry Frequency Predictor",
    page_icon="🏛️",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title("Fundamental Frequency Prediction of Masonry Buildings")

st.write(
    "Enter the geometric and material properties to obtain predictions "
    "from the trained machine-learning models and the explicit equations."
)


# ============================================================
# Valid parameter domain
# ============================================================

st.subheader("Valid parameter ranges")

st.info(
    "The models were developed and evaluated only within the numerical "
    "parameter ranges listed below. Predictions within these bounds "
    "represent interpolation within the investigated numerical domain."
)

range_table = pd.DataFrame(
    {
        "Parameter": [
            "Plan width, B",
            "Plan length, L",
            "Structural height, H",
            "Elastic modulus, E",
            "Wall density, ρ",
        ],
        "Unit": [
            "m",
            "m",
            "m",
            "MPa",
            "kg/m³",
        ],
        "Minimum": [
            valid_ranges["B"][0],
            valid_ranges["L"][0],
            valid_ranges["H"][0],
            valid_ranges["E"][0],
            valid_ranges["rho"][0],
        ],
        "Maximum": [
            valid_ranges["B"][1],
            valid_ranges["L"][1],
            valid_ranges["H"][1],
            valid_ranges["E"][1],
            valid_ranges["rho"][1],
        ],
    }
)

st.dataframe(
    range_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Extrapolation option
# ============================================================

allow_extrapolation = st.checkbox(
    "Allow extrapolation outside the investigated parameter ranges",
    value=False,
)

if allow_extrapolation:
    st.warning(
        "Extrapolation is enabled. Predictions for inputs outside the "
        "investigated parameter ranges have not been validated and "
        "should be interpreted with caution."
    )
else:
    st.caption(
        "Extrapolation is disabled. Input values are restricted to "
        "the parameter ranges used for model development."
    )


# ============================================================
# User inputs
# ============================================================

left, right = st.columns(2)


# ------------------------------------------------------------
# If extrapolation is allowed, do not impose min/max limits.
# Otherwise, restrict values to the investigated domain.
# ------------------------------------------------------------

with left:

    if allow_extrapolation:

        B = st.number_input(
            "Plan width, B (m)",
            value=float(ranges[roles["B"]]["default"]),
            step=0.10,
        )

        L = st.number_input(
            "Plan length, L (m)",
            value=float(ranges[roles["L"]]["default"]),
            step=0.10,
        )

        H = st.number_input(
            "Structural height, H (m)",
            value=float(ranges[roles["H"]]["default"]),
            step=0.10,
        )

    else:

        B = st.number_input(
            "Plan width, B (m)",
            min_value=valid_ranges["B"][0],
            max_value=valid_ranges["B"][1],
            value=float(ranges[roles["B"]]["default"]),
            step=0.10,
        )

        L = st.number_input(
            "Plan length, L (m)",
            min_value=valid_ranges["L"][0],
            max_value=valid_ranges["L"][1],
            value=float(ranges[roles["L"]]["default"]),
            step=0.10,
        )

        H = st.number_input(
            "Structural height, H (m)",
            min_value=valid_ranges["H"][0],
            max_value=valid_ranges["H"][1],
            value=float(ranges[roles["H"]]["default"]),
            step=0.10,
        )


with right:

    if allow_extrapolation:

        E = st.number_input(
            "Elastic modulus, E (MPa)",
            value=float(ranges[roles["E"]]["default"]),
            step=50.0,
        )

        rho = st.number_input(
            "Wall density, ρ (kg/m³)",
            value=float(ranges[roles["rho"]]["default"]),
            step=50.0,
        )

    else:

        E = st.number_input(
            "Elastic modulus, E (MPa)",
            min_value=valid_ranges["E"][0],
            max_value=valid_ranges["E"][1],
            value=float(ranges[roles["E"]]["default"]),
            step=50.0,
        )

        rho = st.number_input(
            "Wall density, ρ (kg/m³)",
            min_value=valid_ranges["rho"][0],
            max_value=valid_ranges["rho"][1],
            value=float(ranges[roles["rho"]]["default"]),
            step=50.0,
        )


# ============================================================
# Check whether current inputs represent extrapolation
# ============================================================

outside_ranges = check_extrapolation(
    B,
    L,
    H,
    E,
    rho,
    valid_ranges,
)


if outside_ranges:

    st.error(
        "⚠️ EXTRAPOLATION WARNING: One or more input parameters fall "
        "outside the numerical domain used to develop and validate the "
        "prediction models."
    )

    warning_rows = []

    parameter_names = {
        "B": "Plan width, B",
        "L": "Plan length, L",
        "H": "Structural height, H",
        "E": "Elastic modulus, E",
        "rho": "Wall density, ρ",
    }

    units = {
        "B": "m",
        "L": "m",
        "H": "m",
        "E": "MPa",
        "rho": "kg/m³",
    }

    for parameter, information in outside_ranges.items():

        warning_rows.append(
            {
                "Parameter": parameter_names[parameter],
                "Input value": information["value"],
                "Valid minimum": information["min"],
                "Valid maximum": information["max"],
                "Unit": units[parameter],
            }
        )

    st.dataframe(
        pd.DataFrame(warning_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The resulting prediction represents extrapolation and has not "
        "been validated by the present numerical database. The result "
        "should therefore be interpreted with caution."
    )


# ============================================================
# Prepare model input
# ============================================================

input_df = pd.DataFrame(
    [
        {
            roles["B"]: B,
            roles["L"]: L,
            roles["H"]: H,
            roles["E"]: E,
            roles["rho"]: rho,
        }
    ],
    columns=metadata["feature_columns"],
)


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Predict",
    type="primary",
):

    rows = []

    # --------------------------------------------------------
    # Machine-learning predictions
    # --------------------------------------------------------

    for target, target_models in models.items():

        for model_name, model in target_models.items():

            prediction = float(
                model.predict(input_df)[0]
            )

            rows.append(
                {
                    "Target": target,
                    "Method": model_name,
                    "Predicted frequency (Hz)": prediction,
                }
            )


    # --------------------------------------------------------
    # Explicit regression equations
    # --------------------------------------------------------

    rows.extend(
        [
            {
                "Target": "f1 (Hz)",
                "Method": "Linear Regression Equation",
                "Predicted frequency (Hz)": linear_formula_f1(
                    B,
                    L,
                    H,
                    E,
                    rho,
                ),
            },
            {
                "Target": "f1 (Hz)",
                "Method": "Symbolic Regression Equation",
                "Predicted frequency (Hz)": symbolic_formula_f1(
                    B,
                    L,
                    H,
                    E,
                    rho,
                ),
            },
        ]
    )


    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results = pd.DataFrame(rows)

    results["Predicted frequency (Hz)"] = (
        results["Predicted frequency (Hz)"]
        .round(4)
    )


    # --------------------------------------------------------
    # Extrapolation flag
    # --------------------------------------------------------

    if outside_ranges:

        st.error(
            "These results correspond to EXTRAPOLATION outside the "
            "validated numerical parameter domain."
        )

        results["Prediction domain"] = "Extrapolation"

    else:

        st.success(
            "All input parameters are within the investigated numerical domain."
        )

        results["Prediction domain"] = "Within investigated range"


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    st.subheader("Prediction results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # f1 comparison
    # --------------------------------------------------------

    f1_results = results[
        results["Target"] == "f1 (Hz)"
    ]

    if not f1_results.empty:

        st.subheader("f1 model comparison")

        st.bar_chart(
            f1_results.set_index("Method")[
                "Predicted frequency (Hz)"
            ]
        )


    # --------------------------------------------------------
    # Uncertainty note
    # --------------------------------------------------------

    st.caption(
        "Prediction intervals are not displayed because the current "
        "models were not developed with a formally calibrated "
        "uncertainty-quantification framework. Cross-validated error "
        "metrics reported in the associated study should be used when "
        "interpreting predictive performance."
    )


# ============================================================
# Equations
# ============================================================

with st.expander("Show equations"):

    st.markdown("**Linear regression**")

    st.latex(
        r"""
        f_1=-1.234+0.233B+0.531L-0.738H+
        3.144\sqrt{\frac{E}{\rho}}
        """
    )


    st.markdown("**Symbolic regression**")

    st.latex(
        r"""
        f_1=
        \sqrt{\frac{E}{\rho}}
        \frac{1.273L/H-0.395}
        {\left(L/(B+5.428)\right)^4+0.278}
        """
    )


    st.caption(
        "B, L, and H are in m; E is in MPa; "
        "ρ is in kg/m³; f1 is in Hz."
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Machine-learning predictions are generated using the final models "
    "fitted to the complete dataset after group-based hyperparameter selection."
)

st.caption(
    "The application is intended for use within the investigated numerical "
    "parameter domain. Extrapolated predictions are explicitly flagged and "
    "should not be interpreted as validated estimates."
)
