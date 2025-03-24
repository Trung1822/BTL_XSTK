# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Đường dẫn file Excel trên máy tính
file_path = r"D:\EXcel\BTL\Data.xlsx"  # Thay đổi đường dẫn nếu tệp nằm ở vị trí khác

# Khu vực 1: Đọc dữ liệu ban đầu
df_raw = pd.read_excel(file_path, sheet_name="Sheet1")
df_raw.columns = df_raw.columns.str.strip()

print("Dữ liệu ban đầu:")
print(df_raw.head())
print("\nThông tin dữ liệu ban đầu:")
print(df_raw.info())

# Khu vực 2: Làm sạch và xử lý dữ liệu
df = df_raw.copy()
df = df.dropna()
df['Thời gian đặt món'] = pd.to_datetime(df['Thời gian đặt món'])

df['Tổng giá trị đơn hàng(VNĐ)'] = df['Tổng giá trị đơn hàng(VNĐ)'].astype(str).str.replace('#', '', regex=False)
df['Tổng giá trị đơn hàng(VNĐ)'] = pd.to_numeric(df['Tổng giá trị đơn hàng(VNĐ)'], errors='coerce')

df['Số lượng món ăn'] = df['Số lượng món ăn'].abs()
df['Tổng giá trị đơn hàng(VNĐ)'] = df['Tổng giá trị đơn hàng(VNĐ)'].abs()

df['Số lượng món ăn'] = pd.to_numeric(df['Số lượng món ăn'], errors='coerce')
df['Tổng giá trị đơn hàng(VNĐ)'] = pd.to_numeric(df['Tổng giá trị đơn hàng(VNĐ)'], errors='coerce')

df = df[(df['Số lượng món ăn'] >= 1) & (df['Số lượng món ăn'] <= 10)]
df = df[df['Tổng giá trị đơn hàng(VNĐ)'] >= 0]

df['Thanh toán'] = df['Thanh toán'].str.strip().str.capitalize()
df['Hình Thức'] = df['Hình Thức'].str.strip().str.capitalize()

df['Giờ cụ thể'] = df['Thời gian đặt món'].dt.hour

def classify_time_slot(hour):
    if 5 <= hour < 11:
        return "Sáng"
    elif 11 <= hour < 17:
        return "Trưa"
    elif 17 <= hour < 21:
        return "Chiều"
    else:
        return "Tối"

df['Khung giờ'] = df['Giờ cụ thể'].apply(classify_time_slot)

print("\nDữ liệu sau khi làm sạch:")
print(df.head())
print("\nThông tin dữ liệu sau khi làm sạch:")
print(df.info())

# Khu vực 3: Khám phá dữ liệu (sử dụng dữ liệu sau khi làm sạch)
time_slot_counts = df['Khung giờ'].value_counts().reindex(['Sáng', 'Trưa', 'Chiều', 'Tối'], fill_value=0)
food_counts = df['Món ăn được đặt'].value_counts()
branch_counts = df['Nhà hàng thực hiện giao dịch( Hà Nội )'].value_counts()
daily_period_food_counts = df.groupby('Khung giờ')['Số lượng món ăn'].sum().reindex(['Sáng', 'Trưa', 'Chiều', 'Tối'], fill_value=0)

# Số lượng món ăn trung bình mỗi đơn hàng theo khung giờ
avg_food_per_order = daily_period_food_counts / time_slot_counts
avg_food_per_order = avg_food_per_order.fillna(0)

# Tách bảng thanh toán
at_store_df = df[df['Hình Thức'] == 'Tại quán']
online_df = df[df['Hình Thức'] == 'Online']
at_store_counts = at_store_df['Thanh toán'].value_counts()
online_counts = online_df['Thanh toán'].value_counts()

print("\nThống kê mô tả (dữ liệu sau khi làm sạch):")
print(df[['Số lượng món ăn', 'Tổng giá trị đơn hàng(VNĐ)']].describe())
print("\nKhung giờ cao điểm (Sáng, Trưa, Chiều, Tối):")
print(time_slot_counts)
print("\nTần suất món ăn:")
print(food_counts)
print("\nSố lượng đơn hàng theo chi nhánh:")
print(branch_counts)
print("\nTổng số lượng món ăn theo khung giờ (Sáng, Trưa, Chiều, Tối):")
print(daily_period_food_counts)
print("\nSố lượng món ăn trung bình mỗi đơn hàng theo khung giờ:")
print(avg_food_per_order)
print("\nPhương thức thanh toán tại quán:")
print(at_store_counts)
print("\nPhương thức thanh toán online:")
print(online_counts)

# Khu vực 4: Trực quan hóa dữ liệu
sns.set(style="whitegrid")

# 4.1. Biểu đồ khung giờ cao điểm
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Khung giờ', y='count', hue='Khung giờ', data=time_slot_counts.reset_index(), palette="Blues_d", legend=False)
plt.title("Khung giờ cao điểm", fontsize=16)
plt.xlabel("Khung giờ", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=0)
plt.savefig(r"D:\EXcel\BTL\time_slot_counts.png")
plt.show()

# 4.2. Biểu đồ tần suất món ăn
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Món ăn được đặt', y='count', hue='Món ăn được đặt', data=food_counts.reset_index(), palette="Greens_d", legend=False)
plt.title("Tần suất món ăn được đặt", fontsize=16)
plt.xlabel("Món ăn", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\food_counts.png")
plt.show()

# 4.3. Biểu đồ phương thức thanh toán tại quán
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Thanh toán', y='count', hue='Thanh toán', data=at_store_counts.reset_index(), palette="Reds_d", legend=False)
plt.title("Phương thức thanh toán tại quán", fontsize=16)
plt.xlabel("Phương thức thanh toán", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\at_store_counts.png")
plt.show()

# 4.4. Biểu đồ phương thức thanh toán online
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Thanh toán', y='count', hue='Thanh toán', data=online_counts.reset_index(), palette="Reds_d", legend=False)
plt.title("Phương thức thanh toán online", fontsize=16)
plt.xlabel("Phương thức thanh toán", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\online_counts.png")
plt.show()

# 4.5. Biểu đồ số lượng giao dịch theo giờ cụ thể
plt.figure(figsize=(12, 6))
hourly_counts = df['Giờ cụ thể'].value_counts().sort_index()
ax = sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o', color='purple')
plt.title("Số lượng giao dịch theo giờ trong ngày", fontsize=16)
plt.xlabel("Giờ cụ thể", fontsize=12)
plt.ylabel("Số lượng giao dịch", fontsize=12)
for i, value in enumerate(hourly_counts.values):
    ax.annotate(f'{value}', (hourly_counts.index[i], value),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\hourly_transaction_counts.png")
plt.show()

# 4.6. Biểu đồ số lượng món ăn theo khung giờ
plt.figure(figsize=(12, 6))
ax = sns.lineplot(x=daily_period_food_counts.index, y=daily_period_food_counts.values, marker='o', color='orange')
plt.title("Số lượng món ăn theo khung giờ trong ngày", fontsize=16)
plt.xlabel("Khung giờ", fontsize=12)
plt.ylabel("Số lượng món ăn", fontsize=12)
for i, value in enumerate(daily_period_food_counts.values):
    ax.annotate(f'{int(value)}', (i, value),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
plt.xticks(range(4), ['Sáng', 'Trưa', 'Chiều', 'Tối'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\daily_period_food_counts.png")
plt.show()

# Khu vực 5: Xây dựng mô hình (Cải thiện hiệu suất)

# 5.1. Mô hình 1: Dự đoán giờ cao điểm (Random Forest Regressor thay vì Linear Regression)
df_time_slot = df.groupby(['Khung giờ', 'Nhà hàng thực hiện giao dịch( Hà Nội )', 'Thanh toán', 'Hình Thức', 'Giờ cụ thể']).size().reset_index(name='Số lượng đơn hàng')

# Mã hóa dữ liệu
le_khung_gio = LabelEncoder()
le_chi_nhanh = LabelEncoder()
le_thanh_toan = LabelEncoder()
le_hinh_thuc = LabelEncoder()

df_time_slot['Khung giờ'] = le_khung_gio.fit_transform(df_time_slot['Khung giờ'])
df_time_slot['Nhà hàng thực hiện giao dịch( Hà Nội )'] = le_chi_nhanh.fit_transform(df_time_slot['Nhà hàng thực hiện giao dịch( Hà Nội )'])
df_time_slot['Thanh toán'] = le_thanh_toan.fit_transform(df_time_slot['Thanh toán'])
df_time_slot['Hình Thức'] = le_hinh_thuc.fit_transform(df_time_slot['Hình Thức'])

# Chuẩn hóa dữ liệu
scaler_time = StandardScaler()
X_time = df_time_slot[['Khung giờ', 'Nhà hàng thực hiện giao dịch( Hà Nội )', 'Thanh toán', 'Hình Thức', 'Giờ cụ thể']]
X_time_scaled = scaler_time.fit_transform(X_time)
y_time = df_time_slot['Số lượng đơn hàng']

# Chia dữ liệu
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time_scaled, y_time, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest Regressor
model_time = RandomForestRegressor(n_estimators=100, random_state=42)
model_time.fit(X_train_time, y_train_time)

# Đánh giá mô hình
y_pred_time = model_time.predict(X_test_time)
print("\nĐánh giá mô hình dự đoán giờ cao điểm (Random Forest Regressor):")
print("Mean Squared Error (MSE):", mean_squared_error(y_test_time, y_pred_time))
print("R2 Score:", r2_score(y_test_time, y_pred_time))

# 5.2. Mô hình 2: Dự đoán món ăn phổ biến (Random Forest Classifier thay vì Logistic Regression)
df_food = df[['Khung giờ', 'Giờ cụ thể', 'Thanh toán', 'Hình Thức', 'Tổng giá trị đơn hàng(VNĐ)', 'Số lượng món ăn', 'Nhà hàng thực hiện giao dịch( Hà Nội )', 'Món ăn được đặt']].copy()

# Mã hóa dữ liệu
le_khung_gio_food = LabelEncoder()
le_thanh_toan_food = LabelEncoder()
le_hinh_thuc_food = LabelEncoder()
le_chi_nhanh_food = LabelEncoder()
le_mon_an = LabelEncoder()

df_food['Khung giờ'] = le_khung_gio_food.fit_transform(df_food['Khung giờ'])
df_food['Thanh toán'] = le_thanh_toan_food.fit_transform(df_food['Thanh toán'])
df_food['Hình Thức'] = le_hinh_thuc_food.fit_transform(df_food['Hình Thức'])
df_food['Nhà hàng thực hiện giao dịch( Hà Nội )'] = le_chi_nhanh_food.fit_transform(df_food['Nhà hàng thực hiện giao dịch( Hà Nội )'])
df_food['Món ăn được đặt'] = le_mon_an.fit_transform(df_food['Món ăn được đặt'])

# Chuẩn hóa dữ liệu
scaler_food = StandardScaler()
X_food = df_food[['Khung giờ', 'Giờ cụ thể', 'Thanh toán', 'Hình Thức', 'Tổng giá trị đơn hàng(VNĐ)', 'Số lượng món ăn', 'Nhà hàng thực hiện giao dịch( Hà Nội )']]
X_food_scaled = scaler_food.fit_transform(X_food)
y_food = df_food['Món ăn được đặt']

# Chia dữ liệu
X_train_food, X_test_food, y_train_food, y_test_food = train_test_split(X_food_scaled, y_food, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest Classifier với Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
model_food = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model_food, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_food, y_train_food)

# Lấy mô hình tốt nhất
best_model_food = grid_search.best_estimator_
print("\nTham số tốt nhất cho Random Forest Classifier:", grid_search.best_params_)

# Đánh giá mô hình
y_pred_food = best_model_food.predict(X_test_food)
print("\nĐánh giá mô hình dự đoán món ăn phổ biến (Random Forest Classifier):")
print("Accuracy Score:", accuracy_score(y_test_food, y_pred_food))
print("\nClassification Report:")
print(classification_report(y_test_food, y_pred_food, target_names=le_mon_an.classes_, zero_division=0))