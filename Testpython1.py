import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Đường dẫn file Excel
file_path = r"D:\EXcel\BTL\Data.xlsx"

# Khu vực 1: Đọc file Excel
# - Đọc dữ liệu từ file Excel (sheet "Sheet1") vào DataFrame
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 1.1. Chuẩn hóa tên cột
# - Loại bỏ khoảng trắng thừa trong tên cột để tránh lỗi
df.columns = df.columns.str.strip()

# In dữ liệu ban đầu để kiểm tra
print("Dữ liệu ban đầu:")
print(df.head())
print("\nThông tin dữ liệu:")
print(df.info())

# Khu vực 2: Làm sạch dữ liệu
# 2.1. Xử lý giá trị trống (NaN)
# - Xóa các dòng có giá trị trống để đảm bảo dữ liệu đầy đủ
df = df.dropna()

# 2.2. Xử lý cột "Thời gian đặt món"
# - Chuyển cột "Thời gian đặt món" thành định dạng datetime để dễ xử lý thời gian
df['Thời gian đặt món'] = pd.to_datetime(df['Thời gian đặt món'])

# 2.3. Xử lý cột "Tổng giá trị đơn hàng(VNĐ)" có ký tự "#"
# - Loại bỏ ký tự "#" và chuyển thành số, nếu không chuyển được thì thành NaN
df['Tổng giá trị đơn hàng(VNĐ)'] = df['Tổng giá trị đơn hàng(VNĐ)'].astype(str).str.replace('#', '', regex=False)
df['Tổng giá trị đơn hàng(VNĐ)'] = pd.to_numeric(df['Tổng giá trị đơn hàng(VNĐ)'], errors='coerce')

# 2.4. Xử lý cột "Số lượng món ăn" và "Tổng giá trị đơn hàng(VNĐ)" có giá trị âm
# - Chuyển các giá trị âm thành giá trị dương (lấy giá trị tuyệt đối)
df['Số lượng món ăn'] = df['Số lượng món ăn'].abs()
df['Tổng giá trị đơn hàng(VNĐ)'] = df['Tổng giá trị đơn hàng(VNĐ)'].abs()

# 2.5. Đảm bảo định dạng số cho các cột
# - Chuyển cột "Số lượng món ăn" và "Tổng giá trị đơn hàng(VNĐ)" thành số, nếu không được thì thành NaN
df['Số lượng món ăn'] = pd.to_numeric(df['Số lượng món ăn'], errors='coerce')
df['Tổng giá trị đơn hàng(VNĐ)'] = pd.to_numeric(df['Tổng giá trị đơn hàng(VNĐ)'], errors='coerce')

# 2.6. Kiểm tra và xử lý outliers
# - Giới hạn số lượng món ăn từ 1 đến 10, tổng giá trị không âm
df = df[(df['Số lượng món ăn'] >= 1) & (df['Số lượng món ăn'] <= 10)]
df = df[df['Tổng giá trị đơn hàng(VNĐ)'] >= 0]

# 2.7. Chuẩn hóa dữ liệu
# - Loại bỏ khoảng trắng và chuẩn hóa chữ hoa/chữ thường cho cột "Thanh toán" và "Hình Thức"
df['Thanh toán'] = df['Thanh toán'].str.strip().str.capitalize()
df['Hình Thức'] = df['Hình Thức'].str.strip().str.capitalize()

# 2.8. Feature Engineering
# - Thêm cột "Giờ cụ thể" (trích xuất giờ từ "Thời gian đặt món")
df['Giờ cụ thể'] = df['Thời gian đặt món'].dt.hour
# - Thêm cột "Khung giờ" (phân loại thành 4 khung giờ)
def classify_time_slot(hour):
    if 1 <= hour <= 6:
        return "1h-6h"
    elif 7 <= hour <= 12:
        return "7h-12h"
    elif 13 <= hour <= 18:
        return "13h-18h"
    else:
        return "19h-24h"
df['Khung giờ'] = df['Giờ cụ thể'].apply(classify_time_slot)

# In dữ liệu sau khi làm sạch để kiểm tra
print("\nDữ liệu sau khi làm sạch:")
print(df.head())
print("\nThông tin dữ liệu sau khi làm sạch:")
print(df.info())

# Khu vực 3: Khám phá dữ liệu
# 3.1. Khung giờ cao điểm
time_slot_counts = df['Khung giờ'].value_counts()

# 3.2. Tần suất món ăn
food_counts = df['Món ăn được đặt'].value_counts()

# 3.3. Loại hình giao dịch
df['Loại hình giao dịch'] = df['Thanh toán'] + " - " + df['Hình Thức']
transaction_counts = df['Loại hình giao dịch'].value_counts()

# 3.4. Phân tích theo chi nhánh
branch_counts = df['Nhà hàng thực hiện giao dịch( Hà Nội )'].value_counts()

# 3.5. Thống kê mô tả
print("\nThống kê mô tả:")
print(df[['Số lượng món ăn', 'Tổng giá trị đơn hàng(VNĐ)']].describe())

# In kết quả phân tích
print("\nKhung giờ cao điểm:")
print(time_slot_counts)
print("\nTần suất món ăn:")
print(food_counts)
print("\nLoại hình giao dịch:")
print(transaction_counts)
print("\nSố lượng đơn hàng theo chi nhánh:")
print(branch_counts)

# Khu vực 4: Trực quan hóa dữ liệu
# - Thiết lập phong cách cho biểu đồ
sns.set(style="whitegrid")

# 4.1. Biểu đồ khung giờ cao điểm
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Khung giờ', y='count', hue='Khung giờ', data=time_slot_counts.reset_index(), palette="Blues_d", legend=False)
plt.title("Khung giờ cao điểm", fontsize=16)
plt.xlabel("Khung giờ", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
# Thêm giá trị trên đầu mỗi cột
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
# Thêm giá trị trên đầu mỗi cột
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\food_counts.png")
plt.show()

# 4.3. Biểu đồ loại hình giao dịch
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Loại hình giao dịch', y='count', hue='Loại hình giao dịch', data=transaction_counts.reset_index(), palette="Reds_d", legend=False)
plt.title("Loại hình giao dịch ưa chuộng", fontsize=16)
plt.xlabel("Loại hình giao dịch", fontsize=12)
plt.ylabel("Số lượng đơn hàng", fontsize=12)
# Thêm giá trị trên đầu mỗi cột
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"D:\EXcel\BTL\transaction_counts.png")
plt.show()

# Khu vực 5: Xây dựng mô hình
# 5.1. Mô hình 1: Dự đoán giờ cao điểm (Hồi quy - Linear Regression)
# - Chuẩn bị dữ liệu: Tạo DataFrame với số lượng đơn hàng theo khung giờ
df_time_slot = df.groupby(['Khung giờ', 'Nhà hàng thực hiện giao dịch( Hà Nội )', 'Thanh toán', 'Hình Thức']).size().reset_index(name='Số lượng đơn hàng')

# Mã hóa các cột phân loại thành số
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
X_time = df_time_slot[['Khung giờ', 'Nhà hàng thực hiện giao dịch( Hà Nội )', 'Thanh toán', 'Hình Thức']]
X_time_scaled = scaler_time.fit_transform(X_time)
y_time = df_time_slot['Số lượng đơn hàng']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time_scaled, y_time, test_size=0.2, random_state=42)

# Huấn luyện mô hình Linear Regression
model_time = LinearRegression()
model_time.fit(X_train_time, y_train_time)

# Dự đoán và đánh giá
y_pred_time = model_time.predict(X_test_time)
print("\nĐánh giá mô hình dự đoán giờ cao điểm (Linear Regression):")
print("Mean Squared Error (MSE):", mean_squared_error(y_test_time, y_pred_time))
print("R2 Score:", r2_score(y_test_time, y_pred_time))

# 5.2. Mô hình 2: Dự đoán món ăn phổ biến (Phân loại - Logistic Regression)
# - Chuẩn bị dữ liệu: Sử dụng các đặc trưng khung giờ, giờ cụ thể, hình thức thanh toán, hình thức nhận hàng, tổng giá trị đơn hàng
df_food = df[['Khung giờ', 'Giờ cụ thể', 'Thanh toán', 'Hình Thức', 'Tổng giá trị đơn hàng(VNĐ)', 'Món ăn được đặt']].copy()

# Mã hóa các cột phân loại thành số
le_khung_gio_food = LabelEncoder()
le_thanh_toan_food = LabelEncoder()
le_hinh_thuc_food = LabelEncoder()
le_mon_an = LabelEncoder()

df_food['Khung giờ'] = le_khung_gio_food.fit_transform(df_food['Khung giờ'])
df_food['Thanh toán'] = le_thanh_toan_food.fit_transform(df_food['Thanh toán'])
df_food['Hình Thức'] = le_hinh_thuc_food.fit_transform(df_food['Hình Thức'])
df_food['Món ăn được đặt'] = le_mon_an.fit_transform(df_food['Món ăn được đặt'])

# Chuẩn hóa dữ liệu
scaler_food = StandardScaler()
X_food = df_food[['Khung giờ', 'Giờ cụ thể', 'Thanh toán', 'Hình Thức', 'Tổng giá trị đơn hàng(VNĐ)']]
X_food_scaled = scaler_food.fit_transform(X_food)
y_food = df_food['Món ăn được đặt']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_food, X_test_food, y_train_food, y_test_food = train_test_split(X_food_scaled, y_food, test_size=0.2, random_state=42)

# Tối ưu hóa tham số cho Logistic Regression bằng Grid Search
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
model_food = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
grid_search = GridSearchCV(model_food, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_food, y_train_food)

# Lấy mô hình tốt nhất
best_model_food = grid_search.best_estimator_
print("\nTham số tốt nhất cho Logistic Regression:", grid_search.best_params_)

# Dự đoán và đánh giá
y_pred_food = best_model_food.predict(X_test_food)
print("\nĐánh giá mô hình dự đoán món ăn phổ biến (Logistic Regression):")
print("Accuracy Score:", accuracy_score(y_test_food, y_pred_food))
print("\nClassification Report:")
print(classification_report(y_test_food, y_pred_food, target_names=le_mon_an.classes_, zero_division=0))