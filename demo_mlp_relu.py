import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn

st.title("⚽ Dự đoán chỉ số tổng quan cầu thủ FIFA")

@st.cache_resource
def load_model_and_preprocessor():
    # Tải preprocessor (bao gồm TargetEncoder và StandardScaler)
    with open('preprocessor.pkl', 'rb') as file:
        encoder, scaler = pickle.load(file)
    
    # Tải mô hình MLP
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim):
            super(MLPRegressor, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
            )
            self.layer3 = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
            )
            self.sc1 = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
            )
            # self.sc2 = nn.Sequential(
            #     nn.Linear(256, 256),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(256),
            # )
            self.fc = nn.Linear(256, 1)
            self.init_weights()

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            prev_x = x
            x = self.layer1(x)
            x += self.sc1(prev_x)
            prev_x = x
            x = self.layer2(x)
            # x += self.sn2(prev_x)
            x += prev_x
            x = self.layer3(x)
            x = self.fc(x)
            return x
    
    model = MLPRegressor(input_dim=52)  # 52 là số đặc trưng đầu vào
    model.load_state_dict(torch.load('mlp_regressor_relu.pth'))
    model.eval()
    return model, encoder, scaler

model, encoder, scaler = load_model_and_preprocessor()

# Danh sách tên cột theo thứ tự chính xác
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

# Tải dữ liệu gốc
@st.cache_resource
def load_data():
    data = pd.read_csv('data.csv')
    return data

data = load_data()

# Thanh bên trái để chọn chức năng
st.sidebar.title("Chọn chức năng")
tab_selected = st.sidebar.selectbox("Chọn:", ["🔍 Chọn cầu thủ trong dữ liệu", "✏️ Điền giá trị cầu thủ"])

if tab_selected == "🔍 Chọn cầu thủ trong dữ liệu":
    # Chọn cầu thủ trong dữ liệu
    st.subheader("Chọn cầu thủ để dự đoán")
    data_display = data.copy()
    data_display['index'] = data_display.index

    # Chọn một mẫu dữ liệu từ danh sách
    selected_index = st.selectbox("Danh sách cầu thủ:", data_display['index'].tolist())

    # Lấy hàng dữ liệu từ DataFrame
    selected_row = data.iloc[selected_index]

    # Hiển thị thông tin cầu thủ
    st.markdown("### 📋 Thông tin cầu thủ")
    st.table(pd.DataFrame(selected_row).T)

    # Dự đoán
    if st.button("Dự đoán"):
        input_df = pd.DataFrame([selected_row.drop(labels=["overall_rating"])])
        # Tiền xử lý dữ liệu
        input_df = input_df[FEATURE_NAMES]
        input_data = encoder.transform(input_df)
        input_data = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        prediction = model(input_tensor).item()
        
        # Hiển thị kết quả
        st.markdown("### 🏆 Kết quả dự đoán")
        st.success(f"**Chỉ số dự đoán (Overall Rating): {prediction:.2f}**")
        st.info(f"**Chỉ số thật (Overall Rating): {selected_row['overall_rating']}**")

elif tab_selected == "✏️ Điền giá trị cầu thủ":
    # Điền giá trị cầu thủ
    st.subheader("Nhập thông tin cầu thủ để dự đoán")
    
    nationality = st.text_input("Quốc tịch (VD: Argentina, Brazil, Germany)")
    age = st.number_input("Tuổi", min_value=15, max_value=50)
    height_cm = st.number_input("Chiều cao (cm)", min_value=100, max_value=250)
    weight_kgs = st.number_input("Cân nặng (kg)", min_value=30, max_value=150)
    preferred_foot = st.selectbox("Chân thuận", ["Left", "Right"])
    weak_foot = st.slider("Độ mạnh của chân không thuận (1-5)", 1, 5)
    skill_moves = st.slider("Kỹ năng di chuyển (1-5)", 1, 5)
    
    # Các kỹ năng chuyên môn
    crossing = st.slider("Tạt bóng", 0, 100)
    finishing = st.slider("Dứt điểm", 0, 100)
    heading_accuracy = st.slider("Đánh đầu chính xác", 0, 100)
    short_passing = st.slider("Chuyền ngắn", 0, 100)
    volleys = st.slider("Vô lê", 0, 100)
    dribbling = st.slider("Dẫn bóng", 0, 100)
    curve = st.slider("Đá cong", 0, 100)
    freekick_accuracy = st.slider("Đá phạt", 0, 100)
    long_passing = st.slider("Chuyền dài", 0, 100)
    ball_control = st.slider("Khống chế bóng", 0, 100)
    acceleration = st.slider("Tăng tốc", 0, 100)
    sprint_speed = st.slider("Tốc độ chạy nước rút", 0, 100)
    agility = st.slider("Khéo léo", 0, 100)
    balance = st.slider("Thăng bằng", 0, 100)
    shot_power = st.slider("Sức mạnh sút", 0, 100)
    jumping = st.slider("Nhảy cao", 0, 100)
    stamina = st.slider("Thể lực", 0, 100)
    strength = st.slider("Sức mạnh", 0, 100)
    long_shots = st.slider("Sút xa", 0, 100)
    aggression = st.slider("Sự quyết liệt", 0, 100)
    interceptions = st.slider("Cắt bóng", 0, 100)
    positioning = st.slider("Duy trì vị trí", 0, 100)
    vision = st.slider("Tầm nhìn", 0, 100)
    penalties = st.slider("Đá penalty", 0, 100)
    marking = st.slider("Kèm người", 0, 100)
    standing_tackle = st.slider("Tranh bóng đứng", 0, 100)
    sliding_tackle = st.slider("Tranh bóng trượt", 0, 100)
    
    # Dạng người
    body_type = st.selectbox("Dáng người", ["Lean", "Normal", "Stocky"])
    body_type_Lean = 1 if body_type == "Lean" else 0
    body_type_Normal = 1 if body_type == "Normal" else 0
    body_type_Stocky = 1 if body_type == "Stocky" else 0
    
    # Vị trí
    positions = st.multiselect(
        "Vị trí chơi", 
        ["positions_CAM", "positions_CB", "positions_CDM", "positions_CF", "positions_CM", 
         "positions_GK", "positions_LB", "positions_LM", "positions_LW", "positions_LWB", 
         "positions_RB", "positions_RM", "positions_RW", "positions_RWB", "positions_ST"]
    )
    positions_dict = {pos: 1 if pos in positions else 0 for pos in [
        "positions_CAM", "positions_CB", "positions_CDM", "positions_CF", "positions_CM", 
        "positions_GK", "positions_LB", "positions_LM", "positions_LW", "positions_LWB", 
        "positions_RB", "positions_RM", "positions_RW", "positions_RWB", "positions_ST"]}

    # Dự đoán
    if st.button("Dự đoán"):
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
        # Tiền xử lý dữ liệu
        input_df = input_df[FEATURE_NAMES]
        input_data_processed = encoder.transform(input_df)
        input_data_processed = scaler.transform(input_data_processed)
        input_tensor = torch.tensor(input_data_processed, dtype=torch.float32)
        prediction = model(input_tensor).item()
        
        st.markdown("### 🏆 Kết quả dự đoán")
        st.success(f"**Chỉ số dự đoán (Overall Rating): {prediction:.2f}**")
