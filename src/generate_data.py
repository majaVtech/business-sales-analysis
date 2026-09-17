import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_ORDERS = 20_000

START_DATE = "2025-01-01"
END_DATE = "2025-12-31" 

dates = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="D"
)

products = {
    "Wireless Mouse": ("Electronics", 24.99),
    "Mechanical Keyboard": ("Electronics", 79.99),
    "Bluetooth Speaker": ("Electronics", 59.99),
    "Smartphone": ("Electronics", 499.99),

    "Coffee Maker": ("Home & Kitchen", 89.99),
    "Air Fryer": ("Home & Kitchen", 119.99),

    "Office Chair": ("Furniture", 189.99),
    "Desk Lamp": ("Furniture", 49.99),

    "Notebook": ("Office Supplies", 5.99),
    "Printer": ("Office Supplies", 149.99),

    "Running Shoes": ("Sports", 89.99),
    "Yoga Mat": ("Sports", 29.99),

    "Skincare Set": ("Beauty", 39.99),
    "Hair Dryer": ("Beauty", 69.99),
}

customer_ids = [
    f"CUST{i :04d}"
    for i in range(1, 501)
]

customer_names = [
    f"Customer {i:04d}"
    for i in range(1, 501)
]
cities = [
    "Banja Luka",
    "Sarajevo",
    "Mostar",
    "Tuzla",
    "Zenica",
    "Bijeljina",
    "Prijedor",
    "Doboj",
    "Brcko",
    "Trebinje"
]

payment_methods = [
    "Credit Card",
    "PayPal",
    "Bank Transfer",
    "Cash on Delivery"
]

order_ids = [
    f"ORD{i:05d}"
    for i in range(1, N_ORDERS + 1)
]

order_dates = np.random.choice(
    dates,
    size=N_ORDERS
)

product_names = list(products.keys())

product_probabilities = [
    0.15,
    0.10,
    0.08,
    0.04,
    0.10,
    0.08,
    0.08,
    0.07,
    0.07,
    0.04,
    0.06,
    0.05,
    0.05,
    0.03
]

selected_products = np.random.choice(
    product_names,
    size=N_ORDERS,
    p=product_probabilities
)

print(sum(product_probabilities))

categories = [
    products[product][0]
    for product in selected_products
]

base_prices = [
    products[product][1]
    for product in selected_products
]

unit_prices = np.round(
    np.array(base_prices) * np.random.uniform(0.95, 1.05, N_ORDERS),
    2
)

quantities = np.random.choice(
    [1, 2, 3, 4, 5],
    size=N_ORDERS,
    p=[0.55, 0.25, 0.12, 0.05, 0.03]
)

discounts = np.random.choice(
    [0, 0.05, 0.10, 0.15, 0.20],
    size=N_ORDERS,
    p=[0.45, 0.25, 0.15, 0.10, 0.05]
)

selected_customer_ids = np.random.choice(
    customer_ids,
    size=N_ORDERS
)

selected_customer_names = [
    f"Customer {customer_id[-4:]}"
    for customer_id in selected_customer_ids
]

selected_cities = np.random.choice(
    cities,
    size=N_ORDERS,
    p=[
        0.22,
        0.20,
        0.12,
        0.10,
        0.08,
        0.07,
        0.06,
        0.05,
        0.06,
        0.04
    ]
)

selected_payment_methods = np.random.choice(
    payment_methods,
    size=N_ORDERS,
    p=[0.45, 0.25, 0.15, 0.15]
)

df = pd.DataFrame({
    "order_id": order_ids,
    "order_date": order_dates,
    "customer_id": selected_customer_ids,
    "customer_name": selected_customer_names,
    "product": selected_products,
    "category": categories,
    "quantity": quantities,
    "unit_price": unit_prices,
    "discount": discounts,
    "city": selected_cities,
    "payment_method": selected_payment_methods
})

df = df.sort_values("order_date").reset_index(drop=True)

missing_city_indices = np.random.choice(
    df.index,
    size=100,
    replace=False
)

df.loc[missing_city_indices, "city"] = np.nan

missing_payment_indices = np.random.choice(
    df.index,
    size=75,
    replace=False
)

df.loc[missing_payment_indices, "payment_method"] = np.nan

category_indices = np.random.choice(
    df.index,
    size=100,
    replace=False
)

df.loc[category_indices, "category"] = (
    df.loc[category_indices, "category"]
    .str.upper()
)

duplicate_rows = df.sample(
    50,
    random_state=42
)

df = pd.concat(
    [df, duplicate_rows],
    ignore_index=True
)

invalid_quantity_indices = np.random.choice(
    df.index,
    size=10,
    replace=False
)

df.loc[
    invalid_quantity_indices,
    "quantity"
] = -1

invalid_price_indices = np.random.choice(
    df.index,
    size=5,
    replace=False
)

df.loc[
    invalid_price_indices,
    "unit_price"
] = -50

output_path = Path("C:\\Users\\Administrator\\Documents\\raw_sales_data.csv")

df.to_csv(
    output_path,
    index=False
)

print(f"Dataset saved to: {output_path}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
