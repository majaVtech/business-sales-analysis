import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)

# ==========================================
# REPORT SETTINGS
# ==========================================

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(REPORT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Professional chart style

MAIN_COLOR = "#1F4E79"
ACCENT_COLOR = "#5B9BD5"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2
})

def format_value(value):
    """Format large business values for chart labels."""

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M KM"

    elif value >= 1_000:
        return f"{value / 1_000:.2f}K KM"

    else:
        return f"{value:.0f} KM"


def axis_formatter(x, pos):
    """Format large numbers on chart axes."""

    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"

    elif x >= 1_000:
        return f"{x / 1_000:.0f}K"

    else:
        return f"{x:.0f}"
    
# ==========================================
# 1. DATA LOADING
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "majaVtech/business-sales-analysis/main/"
    "Data/raw_sales_data.csv"
)


def load_data():
    """Load raw sales data from GitHub."""
    
    df = pd.read_csv(DATA_URL)

    print(f"Raw dataset shape: {df.shape}")

    return df


# ==========================================
# 2. DATA CLEANING
# ==========================================

def clean_data(df):
    """Clean the raw sales dataset."""

    df_clean = df.copy()

    # Remove complete duplicate records
    df_clean = df_clean.drop_duplicates()

    # Remove invalid transactions
    df_clean = df_clean[
        (df_clean["quantity"] >= 0) &
        (df_clean["unit_price"] >= 0)
    ].copy()

    # Standardize category names
    df_clean["category"] = (
        df_clean["category"]
        .str.strip()
        .str.title()
    )

    # Convert order date to datetime
    df_clean["order_date"] = pd.to_datetime(
        df_clean["order_date"],
        errors="coerce"
    )

    print(f"Cleaned dataset shape: {df_clean.shape}")

    return df_clean

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def create_features(df_clean):
    """Create variables required for the analysis."""

    df_clean["revenue"] = (
        df_clean["quantity"]
        * df_clean["unit_price"]
        * (1 - df_clean["discount"])
    )

    df_clean["month"] = (
        df_clean["order_date"].dt.month
    )

    df_clean["month_name"] = (
        df_clean["order_date"].dt.month_name()
    )

    return df_clean

# ==========================================
# 4. KEY BUSINESS KPIs
# ==========================================

def calculate_kpis(df_clean):
    """Calculate the main business KPIs."""

    total_revenue = df_clean["revenue"].sum()
    
    total_orders = df_clean["order_id"].nunique()
    
    units_sold = df_clean["quantity"].sum()
    
    average_order_value = (
        total_revenue / total_orders
    )
    
    average_revenue_per_unit = (
        total_revenue / units_sold
    )

    kpis = {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "units_sold": units_sold,
        "average_order_value": average_order_value,
        "average_revenue_per_unit": average_revenue_per_unit
    }

    return kpis

# ==========================================
# 5. ANALYSIS TABLES
# ==========================================

def calculate_analysis_tables(df_clean):
    """Calculate all tables required for the business report."""

    # --------------------------------------
    # Monthly Sales
    # --------------------------------------

    monthly_sales = (
        df_clean
        .groupby("month")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
    )

    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_sales["month_name"] = (
        monthly_sales["month"]
        .map(dict(enumerate(month_order, start=1)))
    )

    # --------------------------------------
    # Revenue by Category
    # --------------------------------------

    revenue_by_category = (
        df_clean
        .groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    # --------------------------------------
    # Revenue by Product
    # --------------------------------------

    revenue_by_product = (
        df_clean
        .groupby("product")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    # --------------------------------------
    # Top Products by Units Sold
    # --------------------------------------

    top_products_units = (
        df_clean
        .groupby("product")
        .agg(
            units_sold=("quantity", "sum"),
            orders=("order_id", "nunique"),
            revenue=("revenue", "sum")
        )
        .reset_index()
        .sort_values("units_sold", ascending=False)
    )

    # --------------------------------------
    # Customer Analysis
    # --------------------------------------

    customer_analysis = (
        df_clean
        .groupby(["customer_id", "customer_name"])
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    customer_analysis["avg_order_value"] = (
        customer_analysis["revenue"]
        / customer_analysis["orders"]
    )

    # --------------------------------------
    # Revenue by City
    # --------------------------------------

    revenue_by_city = (
        df_clean
        .groupby("city")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    revenue_by_city["avg_order_value"] = (
        revenue_by_city["revenue"]
        / revenue_by_city["orders"]
    )

    # --------------------------------------
    # Payment Method Analysis
    # --------------------------------------

    payment_analysis = (
        df_clean
        .groupby("payment_method")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
    )

    payment_analysis["avg_order_value"] = (
        payment_analysis["revenue"]
        / payment_analysis["orders"]
    )

    payment_analysis = payment_analysis.sort_values(
        "revenue",
        ascending=False
    )

    # --------------------------------------
    # Discount Analysis
    # --------------------------------------

    discount_analysis = (
        df_clean
        .groupby("discount")
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique"),
            units_sold=("quantity", "sum")
        )
        .reset_index()
        .sort_values("discount")
    )

    discount_analysis["avg_order_value"] = (
        discount_analysis["revenue"]
        / discount_analysis["orders"]
    )

    total_revenue = df_clean["revenue"].sum()

    discount_analysis["revenue_share"] = (
        discount_analysis["revenue"]
        / total_revenue
        * 100
    )

    discount_analysis["discount_pct"] = (
        discount_analysis["discount"] * 100
    )

    # --------------------------------------
    # Return all tables
    # --------------------------------------

    return {
        "monthly_sales": monthly_sales,
        "revenue_by_category": revenue_by_category,
        "revenue_by_product": revenue_by_product,
        "top_products_units": top_products_units,
        "customer_analysis": customer_analysis,
        "revenue_by_city": revenue_by_city,
        "payment_analysis": payment_analysis,
        "discount_analysis": discount_analysis
    }

# ==========================================
# 6. CHART GENERATION
# ==========================================

def create_monthly_revenue_chart(analysis):
    """Create monthly revenue chart."""

    monthly_sales = analysis["monthly_sales"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        monthly_sales["month_name"],
        monthly_sales["revenue"],
        marker="o",
        linewidth=2.5,
        color=MAIN_COLOR
    )

    ax.set_title("Monthly Revenue — 2025")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (KM)")

    ax.tick_params(axis="x", rotation=45)

    ax.yaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "monthly_revenue.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_category_revenue_chart(analysis):
    """Create revenue by category chart."""

    revenue_by_category = analysis["revenue_by_category"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        revenue_by_category["category"],
        revenue_by_category["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Revenue by Category — 2025")
    ax.set_xlabel("Revenue (KM)")
    ax.set_ylabel("Category")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        revenue_by_category["revenue"].max() * 1.15
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in revenue_by_category["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "revenue_by_category.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_top_products_revenue_chart(analysis):
    """Create top 10 products by revenue chart."""

    top_products = (
        analysis["revenue_by_product"]
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        top_products["product"],
        top_products["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Top 10 Products by Revenue — 2025")
    ax.set_xlabel("Revenue (KM)")
    ax.set_ylabel("Product")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        top_products["revenue"].max() * 1.15
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in top_products["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "top_products_revenue.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_top_products_units_chart(analysis):
    """Create top 10 products by units sold chart."""

    top_products = (
        analysis["top_products_units"]
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        top_products["product"],
        top_products["units_sold"],
        color=ACCENT_COLOR
    )

    ax.set_title("Top 10 Products by Units Sold — 2025")
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Product")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        top_products["units_sold"].max() * 1.15
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:,.0f}"
            for value in top_products["units_sold"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "top_products_units.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_top_customers_revenue_chart(analysis):
    """Create top 10 customers by revenue chart."""

    top_customers = (
        analysis["customer_analysis"]
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        top_customers["customer_id"],
        top_customers["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Top 10 Customers by Revenue — 2025")
    ax.set_xlabel("Revenue (KM)")
    ax.set_ylabel("Customer")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        top_customers["revenue"].max() * 1.15
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in top_customers["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "top_customers_revenue.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_top_customers_aov_chart(analysis):
    """Create top 10 customers by average order value."""

    top_customers = (
        analysis["customer_analysis"]
        .sort_values(
            "avg_order_value",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        top_customers["customer_id"],
        top_customers["avg_order_value"],
        color=ACCENT_COLOR
    )

    ax.set_title(
        "Top 10 Customers by Average Order Value — 2025"
    )

    ax.set_xlabel("Average Order Value (KM)")
    ax.set_ylabel("Customer")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        top_customers["avg_order_value"].max() * 1.15
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in top_customers["avg_order_value"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "top_customers_aov.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_city_revenue_chart(analysis):
    """Create revenue by city chart."""

    revenue_by_city = analysis["revenue_by_city"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        revenue_by_city["city"],
        revenue_by_city["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Revenue by City — 2025")
    ax.set_xlabel("Revenue (KM)")
    ax.set_ylabel("City")

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        revenue_by_city["revenue"].max() * 1.15
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in revenue_by_city["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "revenue_by_city.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_payment_revenue_chart(analysis):
    """Create revenue by payment method chart."""

    payment_analysis = analysis["payment_analysis"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        payment_analysis["payment_method"],
        payment_analysis["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Revenue by Payment Method — 2025")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Revenue (KM)")

    ax.tick_params(axis="x", rotation=20)

    ax.set_ylim(
        0,
        payment_analysis["revenue"].max() * 1.15
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in payment_analysis["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "revenue_by_payment_method.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

def create_discount_revenue_chart(analysis):
    """Create revenue by discount level chart."""

    discount_analysis = analysis["discount_analysis"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        discount_analysis["discount_pct"].astype(str) + "%",
        discount_analysis["revenue"],
        color=MAIN_COLOR
    )

    ax.set_title("Revenue by Discount Level — 2025")
    ax.set_xlabel("Discount")
    ax.set_ylabel("Revenue (KM)")

    ax.set_ylim(
        0,
        discount_analysis["revenue"].max() * 1.15
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(axis_formatter)
    )

    ax.bar_label(
        bars,
        labels=[
            format_value(value)
            for value in discount_analysis["revenue"]
        ],
        padding=5
    )

    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    output_path = os.path.join(
        CHARTS_DIR,
        "revenue_by_discount.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path

# ==========================================
# PDF SETTINGS
# ==========================================

PDF_PATH = os.path.join(
    REPORT_DIR,
    "Northstar_Retail_Business_Report_2025.pdf"
)

PAGE_WIDTH, PAGE_HEIGHT = A4

# ==========================================
# PDF STYLES
# ==========================================

def create_pdf_styles():
    """Create styles used throughout the PDF report."""

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor(MAIN_COLOR),
            alignment=TA_CENTER,
            spaceAfter=15 * mm
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=8 * mm
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor(MAIN_COLOR),
            spaceBefore=8 * mm,
            spaceAfter=5 * mm
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubsectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor(MAIN_COLOR),
            spaceBefore=5 * mm,
            spaceAfter=3 * mm
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#333333"),
            alignment=TA_LEFT,
            spaceAfter=4 * mm
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#666666"),
            alignment=TA_LEFT
        )
    )

    return styles

def add_page_footer(canvas, doc):
    """Add footer to each PDF page."""

    canvas.saveState()

    canvas.setStrokeColor(
        colors.HexColor("#D9E2F3")
    )

    canvas.line(
        20 * mm,
        15 * mm,
        PAGE_WIDTH - 20 * mm,
        15 * mm
    )

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.setFillColor(
        colors.HexColor("#777777")
    )

    canvas.drawString(
        20 * mm,
        9 * mm,
        "Northstar Retail — Business Sales Analysis 2025"
    )

    canvas.drawRightString(
        PAGE_WIDTH - 20 * mm,
        9 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()

def create_cover_page(story, styles):
    """Create the report cover page."""

    story.append(
        Spacer(1, 45 * mm)
    )

    story.append(
        Paragraph(
            "Northstar Retail",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            "Business Sales Analysis",
            styles["ReportSubtitle"]
        )
    )

    story.append(
        Paragraph(
            "2025",
            styles["ReportSubtitle"]
        )
    )

    story.append(
        Spacer(1, 25 * mm)
    )

    story.append(
        Paragraph(
            "Sales Performance & Business Insights",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    story.append(
        Paragraph(
            "Prepared as a data-driven business performance report "
            "covering sales, products, customers, geographic performance, "
            "payment methods, discounts, and recommended business actions.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 30 * mm)
    )

    story.append(
        Paragraph(
            "Prepared by Maja Divjak",
            styles["SmallText"]
        )
    )

    story.append(
        Paragraph(
            "2025 Sales Analysis",
            styles["SmallText"]
        )
    )

    story.append(
        PageBreak()
    )

def generate_pdf():
    """Generate the Northstar Retail business report."""

    styles = create_pdf_styles()

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    story = []

    create_cover_page(
        story,
        styles
    )

    # --------------------------------------
    # Executive Summary
    # --------------------------------------

    story.append(
        Paragraph(
            "1. Executive Summary",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Northstar Retail generated approximately "
            "<b>3.03 million KM</b> in revenue across "
            "<b>19,985 orders</b> during 2025. "
            "The analysis evaluates overall sales performance, "
            "product and category contribution, customer activity, "
            "geographic performance, payment methods, and discount usage.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    story.append(
        Paragraph(
            "The analysis also identifies key business patterns and "
            "provides measurable recommendations that can support "
            "future sales monitoring and decision-making.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        PageBreak()
    )

    # --------------------------------------
    # Build PDF
    # --------------------------------------

    doc.build(
        story,
        onFirstPage=add_page_footer,
        onLaterPages=add_page_footer
    )

    print(
        f"\nPDF report created successfully:\n{PDF_PATH}"
    )
    
    
# ==========================================
# 7. MAIN
# ==========================================

if __name__ == "__main__":

    df = load_data()

    df_clean = clean_data(df)

    df_clean = create_features(df_clean)

    kpis = calculate_kpis(df_clean)

    analysis = calculate_analysis_tables(df_clean)

    print("\nData preparation completed successfully.")
    print(f"Final dataset shape: {df_clean.shape}")

    print("\nKey Business KPIs:")
    print(f"Total Revenue: {kpis['total_revenue']:,.2f} KM")
    print(f"Total Orders: {kpis['total_orders']:,}")
    print(f"Units Sold: {kpis['units_sold']:,}")
    print(
        f"Average Order Value: "
        f"{kpis['average_order_value']:,.2f} KM"
    )
    print(
        f"Average Revenue per Unit: "
        f"{kpis['average_revenue_per_unit']:,.2f} KM"
    )
    
    print("\nAnalysis tables created:")
    
    for table_name, table in analysis.items():
        print(f"- {table_name}: {table.shape}")
        
    chart_paths = {}
    
    chart_paths["top_customers_revenue"] = (
        create_top_customers_revenue_chart(analysis)
    )

    chart_paths["top_customers_aov"] = (
        create_top_customers_aov_chart(analysis)
    )

    chart_paths["revenue_by_city"] = (
        create_city_revenue_chart(analysis)
    )

    chart_paths["revenue_by_payment_method"] = (
        create_payment_revenue_chart(analysis)
    )

    chart_paths["revenue_by_discount"] = (
        create_discount_revenue_chart(analysis)
    )

    chart_paths["monthly_revenue"] = (
        create_monthly_revenue_chart(analysis)
    )

    chart_paths["revenue_by_category"] = (
        create_category_revenue_chart(analysis)
    )

    chart_paths["top_products_revenue"] = (
        create_top_products_revenue_chart(analysis)
    )

    chart_paths["top_products_units"] = (
        create_top_products_units_chart(analysis)
    )

    print("\nCharts created:")

    for chart_name, chart_path in chart_paths.items():
        print(f"- {chart_name}: {chart_path}")
        
    generate_pdf()