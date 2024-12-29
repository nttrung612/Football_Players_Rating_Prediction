import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn

# Main title
st.title("⚽ FIFA Player Overall Rating Prediction")

# Load MLP model from .pth file
@st.cache_resource
def load_model_and_preprocessor():
    # Load preprocessor (including TargetEncoder and StandardScaler)
    with open('preprocessor.pkl', 'rb') as file:
        encoder, scaler = pickle.load(file)

    # Define the MLP model
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim):
            super(MLPRegressor, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
            )
            self.layer3 = nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
            )
            self.sc1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
            )
            self.sc2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.BatchNorm1d(256),
            )
            self.fc = nn.Linear(256, 1)
            self.init_weights()

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            prev_x = x
            x = self.layer1(x)
            x += self.sc1(prev_x)
            prev_x = x
            x = self.layer2(x)
            # x += self.sc2(prev_x)
            x += prev_x
            x = self.layer3(x)
            x = self.fc(x)
            return x

    model = MLPRegressor(input_dim=52)  # 52 features as input
    model.load_state_dict(torch.load('mlp_regressor.pth'))
    model.eval()
    return model, encoder, scaler

model, encoder, scaler = load_model_and_preprocessor()

# Correct feature names order
FEATURE_NAMES = [
    "age", "height_cm", "weight_kgs", "nationality", "preferred_foot", "weak_foot(1-5)", "skill_moves(1-5)",
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve",
    "freekick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility",
    "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression",
    "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle",
    "body_type_Lean", "body_type_Normal", "body_type_Stocky", "positions_CAM", "positions_CB",
    "positions_CDM", "positions_CF", "positions_CM", "positions_GK", "positions_LB", "positions_LM",
    "positions_LW", "positions_LWB", "positions_RB", "positions_RM", "positions_RW", "positions_RWB",
    "positions_ST"
]

# Load original and processed datasets
@st.cache_resource
def load_data():
    original_data = pd.read_csv('full_data.csv')
    data = pd.read_csv('data.csv')
    return original_data, data

original_data, data = load_data()

# Sidebar menu to select functionality
st.sidebar.title("Select functionality")
tab_selected = st.sidebar.selectbox("Choose:", ["🔍 Select player from data", "✏️ Enter player attributes"])

if tab_selected == "🔍 Select player from data":
    # Select player from the data
    st.subheader("Select a player to predict")
    data_display = original_data.copy()
    data_display['index'] = data_display.index

    # Select a row from the dataset
    selected_index = st.selectbox("Player list:", data_display['index'].tolist(), format_func=lambda x: data_display.loc[x, 'name'])

    # Get the selected player's row
    selected_row = original_data.iloc[selected_index]

    # Display player information
    st.markdown("### 📋 Player Information")
    st.table(pd.DataFrame(selected_row).T)

    # Make a prediction
    if st.button("Predict"):
        input_df = pd.DataFrame([data.iloc[selected_index].drop(labels=["overall_rating"])])
        # Preprocess the data
        input_df = input_df[FEATURE_NAMES]
        input_data = encoder.transform(input_df)
        input_data = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        prediction = model(input_tensor).item()

        # Display results
        st.markdown("### 🏆 Prediction Result")
        st.success(f"**Predicted Overall Rating: {prediction:.2f}**")
        st.info(f"**Actual Overall Rating: {selected_row['overall_rating']}**")

elif tab_selected == "✏️ Enter player attributes":
    # Enter player attributes manually
    st.subheader("Enter player attributes to predict")

    nationality = st.text_input("Nationality (e.g., Argentina, Brazil, Germany)")
    age = st.number_input("Age", min_value=15, max_value=50)
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250)
    weight_kgs = st.number_input("Weight (kg)", min_value=30, max_value=150)
    preferred_foot = st.selectbox("Preferred Foot", ["Left", "Right"])
    weak_foot = st.slider("Weak Foot Rating (1-5)", 1, 5)
    skill_moves = st.slider("Skill Moves (1-5)", 1, 5)

    # Technical attributes
    crossing = st.slider("Crossing", 0, 100)
    finishing = st.slider("Finishing", 0, 100)
    heading_accuracy = st.slider("Heading Accuracy", 0, 100)
    short_passing = st.slider("Short Passing", 0, 100)
    volleys = st.slider("Volleys", 0, 100)
    dribbling = st.slider("Dribbling", 0, 100)
    curve = st.slider("Curve", 0, 100)
    freekick_accuracy = st.slider("Free Kick Accuracy", 0, 100)
    long_passing = st.slider("Long Passing", 0, 100)
    ball_control = st.slider("Ball Control", 0, 100)
    acceleration = st.slider("Acceleration", 0, 100)
    sprint_speed = st.slider("Sprint Speed", 0, 100)
    agility = st.slider("Agility", 0, 100)
    balance = st.slider("Balance", 0, 100)
    shot_power = st.slider("Shot Power", 0, 100)
    jumping = st.slider("Jumping", 0, 100)
    stamina = st.slider("Stamina", 0, 100)
    strength = st.slider("Strength", 0, 100)
    long_shots = st.slider("Long Shots", 0, 100)
    aggression = st.slider("Aggression", 0, 100)
    interceptions = st.slider("Interceptions", 0, 100)
    positioning = st.slider("Positioning", 0, 100)
    vision = st.slider("Vision", 0, 100)
    penalties = st.slider("Penalties", 0, 100)
    marking = st.slider("Marking", 0, 100)
    standing_tackle = st.slider("Standing Tackle", 0, 100)
    sliding_tackle = st.slider("Sliding Tackle", 0, 100)

    # Physical attributes
    body_type = st.selectbox("Body Type", ["Lean", "Normal", "Stocky"])
    body_type_Lean = 1 if body_type == "Lean" else 0
    body_type_Normal = 1 if body_type == "Normal" else 0
    body_type_Stocky = 1 if body_type == "Stocky" else 0

    # Playing positions
    positions = st.multiselect(
        "Playing Positions",
        ["positions_CAM", "positions_CB", "positions_CDM", "positions_CF", "positions_CM",
         "positions_GK", "positions_LB", "positions_LM", "positions_LW", "positions_LWB",
         "positions_RB", "positions_RM", "positions_RW", "positions_RWB", "positions_ST"]
    )
    positions_dict = {pos: 1 if pos in positions else 0 for pos in [
        "positions_CAM", "positions_CB", "positions_CDM", "positions_CF", "positions_CM",
        "positions_GK", "positions_LB", "positions_LM", "positions_LW", "positions_LWB",
        "positions_RB", "positions_RM", "positions_RW", "positions_RWB", "positions_ST"]}

    # Make a prediction
    if st.button("Predict"):
        input_data = {
            'nationality': nationality, 'age': age, 'height_cm': height_cm, 'weight_kgs': weight_kgs,
            'preferred_foot': 1 if preferred_foot == "Right" else 0,
            'weak_foot(1-5)': weak_foot, 'skill_moves(1-5)': skill_moves, 'crossing': crossing,
            'finishing': finishing, 'heading_accuracy': heading_accuracy, 'short_passing': short_passing,
            'volleys': volleys, 'dribbling': dribbling, 'curve': curve, 'freekick_accuracy': freekick_accuracy,
            'long_passing': long_passing, 'ball_control': ball_control, 'acceleration': acceleration,
            'sprint_speed': sprint_speed, 'agility': agility, 'balance': balance, 'shot_power': shot_power,
            'jumping': jumping, 'stamina': stamina, 'strength': strength, 'long_shots': long_shots,
            'aggression': aggression, 'interceptions': interceptions, 'positioning': positioning,
            'vision': vision, 'penalties': penalties, 'marking': marking, 'standing_tackle': standing_tackle,
            'sliding_tackle': sliding_tackle, 'body_type_Lean': body_type_Lean,
            'body_type_Normal': body_type_Normal, 'body_type_Stocky': body_type_Stocky,
            **positions_dict
        }

        input_df = pd.DataFrame([input_data])
        # Preprocess the data
        input_df = input_df[FEATURE_NAMES]
        input_data_processed = encoder.transform(input_df)
        input_data_processed = scaler.transform(input_data_processed)
        input_tensor = torch.tensor(input_data_processed, dtype=torch.float32)
        prediction = model(input_tensor).item()

        st.markdown("### 🏆 Prediction Result")
        st.success(f"**Predicted Overall Rating: {prediction:.2f}**")
