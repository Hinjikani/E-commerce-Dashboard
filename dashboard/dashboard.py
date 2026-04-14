import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency

sns.set(style='dark')

# Menyiapkan Dataframe
# Menghapus duplicate pada data payments
def drop_payment_duplicate(df):
    df_cleaned = df.drop_duplicates(subset=['order_id'])
    return df_cleaned

# Create_monthly_orders untuk menyiapkan monthly_orders_df

def create_monthly_orders_df(df):
    monthly_orders_df = df.resample(rule='ME', on='order_purchase_timestamp').agg({
        "price":"sum",
        "order_id":"nunique"
    })
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "price": "revenue",
        "order_id": "orders"
    }, inplace=True)

    return monthly_orders_df

# Create_shipping_stats untuk menyiapkan shipping_stats_df
def create_shipping_stats_df(df):
    df = drop_payment_duplicate(df)
    shipping_stats_df = df['delivery_time'].describe()
    return shipping_stats_df

# Create_payment_method untuk menyiapkan payment_method_df
def create_payment_method_df(df):
    payment_method_df = df['payment_type'].value_counts().reset_index()
    payment_method_df.columns = ['payment_type', 'order_count']
    return payment_method_df

# Create_category_sales untuk menyiapkan category_sales_df
def create_category_sales_df(df):
    category_sales_df = df.groupby(by='product_category_name').agg({
        'order_id':'nunique',
        'price':'sum'
    }).reset_index()
    category_sales_df.rename(columns={
        'product_category_name': 'category',
        'order_id':'sales',
        'price':'revenue'
    }, inplace=True)
    return category_sales_df

# Create_review_summary untuk menyiapkan review_summary_df
def create_review_summary_df(df):
    df = drop_payment_duplicate(df)
    review_summary_df = df.groupby('review_status').order_id.nunique().reset_index()
    review_summary_df.rename(columns={
        'order_id':'count'
    }, inplace=True)
    return review_summary_df

# Create_city_sales untuk menyiapkan city_sales_df
def create_city_sales_df(df):
    city_sales_df = df.groupby('customer_city').agg({
        'order_id': 'nunique',
        'price': 'sum'
    })
    city_sales_df.rename(columns={
        'order_id': 'counts',
        'price': 'revenue'
    }, inplace=True)
    return city_sales_df

# Create_state_sales untuk menyiapkan state_sales_df
def create_state_sales_df(df):
    state_sales_df = df.groupby('customer_state').agg({
        'order_id':'nunique',
        'price': 'sum'
    })
    state_sales_df.rename(columns={
        'order_id': 'counts',
        'price': 'revenue'
    }, inplace=True)
    return state_sales_df

def create_RFM_df(df):
    RFM_df = df.groupby(by='customer_id', as_index=False).agg({
    "order_purchase_timestamp": 'max',
    'order_id': 'nunique',
    'price': 'sum'
    })

    RFM_df.columns = ['customer_id', 'latest_order_timestamp', 'frequency', 'monetary']
    RFM_df["latest_order_timestamp"] = RFM_df["latest_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    RFM_df["recency"] = RFM_df["latest_order_timestamp"].apply(lambda x: (recent_date - x).days)
    RFM_df.drop("latest_order_timestamp", axis=1, inplace=True)

    RFM_df['r_rank'] = RFM_df['recency'].rank(ascending=False)
    RFM_df['f_rank'] = RFM_df['frequency'].rank(ascending=True)
    RFM_df['m_rank'] = RFM_df['monetary'].rank(ascending=True)

    # Melakukan normalisasi rank
    RFM_df['r_rank_norm'] = (RFM_df['r_rank']/RFM_df['r_rank'].max())*100
    RFM_df['f_rank_norm'] = (RFM_df['f_rank']/RFM_df['f_rank'].max())*100
    RFM_df['m_rank_norm'] = (RFM_df['m_rank']/RFM_df['m_rank'].max())*100
    
    RFM_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)
    # Membuat RFM Scoring dengan 20% Recency, 10% Frequency, dan 70% Monetary
    RFM_df['RFM_score'] = 0.2*RFM_df['r_rank_norm']+0.1 * RFM_df['f_rank_norm']+0.7 * RFM_df['m_rank_norm']
    RFM_df['RFM_score'] *= 0.05
    RFM_df = RFM_df.round(2)

    # Melakukan segmentasi data
    RFM_df["customer_segment"] = np.where(
    RFM_df['RFM_score'] > 4, "Top customer", (np.where(
        RFM_df['RFM_score'] > 3.3, "High value customer",(np.where(
            RFM_df['RFM_score'] > 2.3, "Medium value customer", np.where(
                RFM_df['RFM_score'] > 0.5, 'Low value customer', 'Lost customer'))))))
    return RFM_df
def create_customer_segment_df(df):
    df = create_RFM_df(df)
    customer_segment_df = df.groupby('customer_segment', as_index=False).customer_id.nunique()

    customer_segment_df.rename(columns={
        'customer_id': 'count'
    }, inplace=True)
    return customer_segment_df

# Mengurutkan data berdasarkan order_purchase_timestamp
try:
    all_df = pd.read_csv('main_data.csv')
except:
    all_df = pd.read_csv('dashboard/main_data.csv')

all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)
datetime_columns = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "shipping_limit_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])


# Membuat komponen filter
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.header("Filters")

    # Mengambil start_date & end_date dari date_input   
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df['order_purchase_timestamp'] >= str(start_date)) & 
                (all_df['order_purchase_timestamp'] <= str(end_date))]

monthly_orders_df = create_monthly_orders_df(main_df)
shipping_stats_df = create_shipping_stats_df(main_df)
payment_method_df = create_payment_method_df(main_df)
category_sales_df = create_category_sales_df(main_df)
review_summary_df = create_review_summary_df(main_df)
city_sales_df = create_city_sales_df(main_df)
state_sales_df = create_state_sales_df(main_df)
RFM_df = create_RFM_df(main_df)
customer_segment_df = create_customer_segment_df(main_df)

# Melengkapi dashboard dengan visualisasi data
st.header('E-Commerce Dashboard')

# Dashboard Monthly Revenue
st.subheader('Monthly Revenue')
total_revenue = format_currency(monthly_orders_df.revenue.sum(), "BRL", locale='es_CO') 
st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    monthly_orders_df["order_purchase_timestamp"],
    monthly_orders_df["revenue"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='y',
    linestyle='--',
    alpha=0.45)
ax.ticklabel_format(style='plain', axis='y')

st.pyplot(fig)

# Dashboard Monthly Orders
st.subheader('Monthly Orders')
col1, col2 = st.columns(2)

total_orders = monthly_orders_df.orders.sum()
st.metric("Total Orders", value=total_orders)
 
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    monthly_orders_df["order_purchase_timestamp"],
    monthly_orders_df["orders"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='y',
    linestyle='--',
    alpha=0.45)

st.pyplot(fig)

# Dashboard Delivery Time Performance
st.header('Delivery Time Performance')
col1, col2 = st.columns(2)

with col1:
    st.metric(label='Average Time', value=f"{shipping_stats_df['mean'].round(2)} Days")
with col2:
    st.metric(label="Standard Deviation", value=f"{shipping_stats_df['std'].round(2)}")
st.subheader('Delivery Time Metrics')
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Min", value=f"{shipping_stats_df['min'].round(2)}")
with col2:
    st.metric(label="25% Percentile", value=f"{shipping_stats_df['25%'].round(2)}")
with col3:
    st.metric(label="50% Percentile", value=f"{shipping_stats_df['50%'].round(2)}")
with col4:
    st.metric(label="75% Percentile", value=f"{shipping_stats_df['75%'].round(2)}")
with col5:
    st.metric(label="Max", value=f"{shipping_stats_df['max'].round(2)}")

# Dashboard Orders Payment Method
st.header('Orders Payment Method')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label='Credit Card', value=payment_method_df[payment_method_df['payment_type'] == 'credit_card']['order_count'])
with col2:
    st.metric(label='Boleto', value=payment_method_df[payment_method_df['payment_type'] == 'boleto']['order_count'])
with col3:
    st.metric(label='Debit_card', value=payment_method_df[payment_method_df['payment_type'] == 'debit_card']['order_count'])
with col4:
    st.metric(label='Voucher', value=payment_method_df[payment_method_df['payment_type'] == 'voucher']['order_count'])

fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(
    x='payment_type',
    y='order_count',
    data=payment_method_df.sort_values(by='order_count', ascending=False),
    palette=["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
)

ax.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=20)
ax.tick_params(axis='x', labelcolor='#FFFFFF', labelsize=20)
ax.set(xlabel=None, ylabel=None)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
st.pyplot(fig)

# Dashboard Category Sales
st.header('Category Sales')
st.subheader('Best Products')
fig, ax = plt.subplots(figsize=(16,8))
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x='sales',
    y='category',
    data=category_sales_df.sort_values(by='sales', ascending=False).head(5),
    palette=colors
    )
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Product Sales",color='#ffffff', loc="center", fontsize=30)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
for container in ax.containers:
    ax.bar_label(container, padding=5, color='white')

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(
    x='revenue',
    y='category',
    data=category_sales_df.sort_values(by='revenue', ascending=False).head(5),
    palette=colors
    )
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Product Revenue", color='#ffffff', loc="center", fontsize=30)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
ax.ticklabel_format(style='plain', axis='x')
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)

st.pyplot(fig)

st.subheader('Worst Products')
fig, ax = plt.subplots(figsize=(16,8))
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x='sales',
    y='category',
    data=category_sales_df.sort_values(by='sales', ascending=True).head(5),
    palette=colors
    )
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Product Sales",color='#ffffff', loc="center", fontsize=30)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
for container in ax.containers:
    ax.bar_label(container, padding=5, color='white')

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(
    x='revenue',
    y='category',
    data=category_sales_df.sort_values(by='revenue', ascending=True).head(5),
    palette=colors
    )
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Product Revenue", color='#ffffff', loc="center", fontsize=30)
ax.tick_params(axis='y', labelcolor='#ffffff', labelsize=20)
ax.tick_params(axis='x', labelcolor='#ffffff', labelsize=15)
ax.ticklabel_format(style='plain', axis='x')
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)

st.pyplot(fig)

# Dashboard Review Performance
st.header('Review Performance')
fig, ax = plt.subplots(figsize=(4, 4))
pie_colors = ["#85FF6A", "#FDFF6C", "#FF7878" ]
patches, texts, autotexts = plt.pie(
    review_summary_df['count'],
    labels=review_summary_df['review_status'],
    autopct='%1.1f%%',
    colors=pie_colors,
    radius=1
)

for text in texts:
    text.set_color('white')
    text.set_fontsize(8)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_weight('bold')
    autotext.set_fontsize(8)

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

st.pyplot(fig, bbox_inches='tight', use_container_width=False)

st.header("Lokasi dengan Penjualan Tertinggi")
st.subheader('State dengan Penjualan Tertinggi')

# Dashboard Penjualan & Revenue per State
st.markdown("#### State berdasarkan Jumlah Order")
fig, ax = plt.subplots (figsize=(16, 8))
sns.barplot(
    x='counts',
    y='customer_state',
    data=state_sales_df.sort_values(by='counts', ascending=False).head(5),
    palette=colors
)
ax.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=20)
ax.tick_params(axis='x', labelcolor='#FFFFFF', labelsize=20)
ax.set(xlabel=None, ylabel=None)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
st.pyplot(fig)

st.markdown("#### State berdasarkan Jumlah Revenue")
fig, ax = plt.subplots (figsize=(16, 8))
sns.barplot(
    x='revenue',
    y='customer_state',
    data=state_sales_df.sort_values(by='revenue', ascending=False).head(5),
    palette=colors
)
ax.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=20)
ax.tick_params(axis='x', labelcolor='#FFFFFF', labelsize=20)
ax.set(xlabel=None, ylabel=None)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.ticklabel_format(style='plain', axis='x')
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
st.pyplot(fig)

# Dashboard Penjualan & Revenue per kota
st.subheader('Kota dengan Penjualan Tertinggi')

st.markdown("#### Kota berdasarkan Jumlah Orders")
fig, ax = plt.subplots (figsize=(16, 8))
sns.barplot(
    x='counts',
    y='customer_city',
    data=city_sales_df.sort_values(by='counts', ascending=False).head(5),
    palette=colors
)
ax.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=20)
ax.tick_params(axis='x', labelcolor='#FFFFFF', labelsize=20)
ax.set(xlabel=None, ylabel=None)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
st.pyplot(fig)

st.markdown("#### Kota berdasarkan Jumlah Revenue")
fig, ax = plt.subplots (figsize=(16, 8))
sns.barplot(
    x='revenue',
    y='customer_city',
    data=city_sales_df.sort_values(by='revenue', ascending=False).head(5),
    palette=colors
)
ax.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=20)
ax.tick_params(axis='x', labelcolor='#FFFFFF', labelsize=20)
ax.set(xlabel=None, ylabel=None)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.ticklabel_format(style='plain', axis='x')
ax.grid(
    visible=True,
    axis='x',
    linestyle='--',
    alpha=0.45)
st.pyplot(fig)

# RFM Analysis
st.subheader('RFM Analysis')

st.markdown('#### Recency')
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(
    x='recency',
    y='customer_id',
    data=RFM_df.sort_values(by="recency", ascending=True).head(5),
    palette=["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
)
ax.set(xlabel=None, ylabel=None)
ax.set_title("By Recency (days)", loc="center", fontsize=18)
ax.tick_params(axis ='x', labelsize=15)
st.pyplot(fig)

st.markdown('#### Frequency')
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(y="customer_id",
            x="frequency",
            data=RFM_df.sort_values(by="frequency", ascending=False).head(5),
            hue='customer_id')
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("By Frequency", loc="center", fontsize=18)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

st.markdown('#### Monetary')
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(y="customer_id",
            x="monetary",
            data=RFM_df.sort_values(by="monetary", ascending=False).head(7),
            hue='customer_id')
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("By Monetary", loc="center", fontsize=18)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

st.markdown('#### Segmentation')
col1, col2, col3, col4, col5 = st.columns(5)



with col1:
    st.metric(label = 'Top Value Customer', value=f"{customer_segment_df.loc[customer_segment_df['customer_segment'] == 'Top customer', 'count'].item()}")
with col2:
    st.metric(label = 'High Value Customer', value=f"{customer_segment_df.loc[customer_segment_df['customer_segment'] == 'High value customer', 'count'].item()}")
with col3:
    st.metric(label = 'Medium Value Customer', value=f"{customer_segment_df.loc[customer_segment_df['customer_segment'] == 'Medium value customer', 'count'].item()}")
with col4:
    st.metric(label = 'Low Value Customer', value=f"{customer_segment_df.loc[customer_segment_df['customer_segment'] == 'Low value customer', 'count'].item()}")
with col5:
    st.metric(label = 'Lost Customer', value=f"{customer_segment_df.loc[customer_segment_df['customer_segment'] == 'Lost customer', 'count'].item()}")

fig, ax = plt.subplots(figsize=(16, 8))
colors_ = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(
    x="count", 
    y="customer_segment",
    data=customer_segment_df.sort_values(by="count", ascending=False),
    palette=colors_
)
plt.title("Number of Customer for Each Segment", loc="center", fontsize=15)
plt.ylabel(None)
plt.xlabel(None)
plt.tick_params(axis='y', labelsize=12)
st.pyplot(fig)