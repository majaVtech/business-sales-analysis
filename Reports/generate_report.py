import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from reportlab.lib import colors
from reportlab.lib import styles
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
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

def create_kpi_section(story, styles, kpis):
    """Create KPI summary cards for the report."""

    story.append(
        Paragraph(
            "2. Key Business KPIs",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "The following indicators provide a high-level overview "
            "of Northstar Retail's sales performance during 2025.",
            styles["BodyTextCustom"]
        )
    )

    kpi_data = [
        [
            Paragraph("<b>Total Revenue</b>", styles["SmallText"]),
            Paragraph("<b>Total Orders</b>", styles["SmallText"]),
            Paragraph("<b>Units Sold</b>", styles["SmallText"]),
        ],
        [
            Paragraph(
                f"<b>{kpis['total_revenue']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['total_orders']:,}</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['units_sold']:,}</b>",
                styles["BodyTextCustom"]
            ),
        ],
        [
            Paragraph("<b>Average Order Value</b>", styles["SmallText"]),
            Paragraph("<b>Revenue per Unit</b>", styles["SmallText"]),
            Paragraph("<b>Analysis Year</b>", styles["SmallText"]),
        ],
        [
            Paragraph(
                f"<b>{kpis['average_order_value']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['average_revenue_per_unit']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                "<b>2025</b>",
                styles["BodyTextCustom"]
            ),
        ]
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm
        ]
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D9E2F3")
            ),
            (
                "BACKGROUND",
                (0, 2),
                (-1, 2),
                colors.HexColor("#D9E2F3")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor(MAIN_COLOR)
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#B4C7E7")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(kpi_table)

    story.append(
        Spacer(1, 10 * mm)
    )
    
def generate_pdf(kpis, analysis, chart_paths):
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

    # --------------------------------------
    # Cover Page
    # --------------------------------------

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
    # Key Business KPIs
    # --------------------------------------

    create_kpi_section(
        story,
        styles,
        kpis
    )

    story.append(
        PageBreak()
    )
    
    # --------------------------------------
    # Sales Performance
    # --------------------------------------

    create_sales_performance_section(
        story,
        styles,
        chart_paths,
        analysis
    )
    
    # --------------------------------------
    # Product & Category Analysis
    # --------------------------------------

    create_product_analysis_section(
        story,
        styles,
        chart_paths,
        analysis
    )
    
    # --------------------------------------
    # Customer Analysis
    # --------------------------------------

    create_customer_analysis_section(
        story,
        styles,
        chart_paths,
        analysis
    )
    
    # --------------------------------------
    # Geographic Analysis
    # --------------------------------------

    create_geographic_analysis_section(
        story,
        styles,
        chart_paths,
        analysis
    )
    
    # --------------------------------------
    # Discount & Payment Analysis
    # --------------------------------------

    create_discount_payment_analysis_section(
        story,
        styles,
        chart_paths,
        analysis
    )
    
    # --------------------------------------
    # Business Insights
    # --------------------------------------

    create_business_insights_section(
        story,
        styles,
        kpis,
        analysis
    )
    
    # --------------------------------------
    # Recommendations
    # --------------------------------------

    create_recommendations_section(
        story,
        styles
    )
    
    # --------------------------------------
    # KPI Targets & Next Steps
    # --------------------------------------

    create_kpi_targets_section(
        story,
        styles,
        kpis
    )
    
    # --------------------------------------
    # Final Conclusion
    # --------------------------------------

    create_final_conclusion_section(
        story,
        styles,
        kpis,
        analysis
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

def create_kpi_section(story, styles, kpis):
    """Create KPI summary cards for the report."""

    story.append(
        Paragraph(
            "2. Key Business KPIs",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "The following indicators provide a high-level overview "
            "of Northstar Retail's sales performance during 2025.",
            styles["BodyTextCustom"]
        )
    )

    kpi_data = [
        [
            Paragraph("<b>Total Revenue</b>", styles["SmallText"]),
            Paragraph("<b>Total Orders</b>", styles["SmallText"]),
            Paragraph("<b>Units Sold</b>", styles["SmallText"]),
        ],
        [
            Paragraph(
                f"<b>{kpis['total_revenue']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['total_orders']:,}</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['units_sold']:,}</b>",
                styles["BodyTextCustom"]
            ),
        ],
        [
            Paragraph("<b>Average Order Value</b>", styles["SmallText"]),
            Paragraph("<b>Revenue per Unit</b>", styles["SmallText"]),
            Paragraph("<b>Analysis Year</b>", styles["SmallText"]),
        ],
        [
            Paragraph(
                f"<b>{kpis['average_order_value']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                f"<b>{kpis['average_revenue_per_unit']:,.2f} KM</b>",
                styles["BodyTextCustom"]
            ),
            Paragraph(
                "<b>2025</b>",
                styles["BodyTextCustom"]
            ),
        ]
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm
        ]
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D9E2F3")
            ),
            (
                "BACKGROUND",
                (0, 2),
                (-1, 2),
                colors.HexColor("#D9E2F3")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor(MAIN_COLOR)
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#B4C7E7")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(kpi_table)

    story.append(
        Spacer(1, 10 * mm)
    )

def create_sales_performance_section(story, styles, chart_paths, analysis):
    """Create the sales performance section of the report."""

    story.append(
        Paragraph(
            "3. Sales Performance",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Sales performance remained relatively stable throughout "
            "2025, with monthly revenue showing moderate variation. "
            "Category-level analysis indicates that revenue is "
            "concentrated in a smaller number of business categories.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Monthly Revenue
    # --------------------------------------

    story.append(
        Paragraph(
            "Monthly Revenue",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["monthly_revenue"],
            width=165 * mm,
            height=82 * mm
        )
    )

    monthly_sales = analysis["monthly_sales"]

    best_month = monthly_sales.loc[
        monthly_sales["revenue"].idxmax()
    ]

    lowest_month = monthly_sales.loc[
        monthly_sales["revenue"].idxmin()
    ]

    story.append(
        Paragraph(
            f"<b>Key observation:</b> Revenue was highest in "
            f"<b>{best_month['month_name']}</b> at "
            f"<b>{format_value(best_month['revenue'])}</b>, "
            f"while the lowest monthly revenue was recorded in "
            f"<b>{lowest_month['month_name']}</b> at "
            f"<b>{format_value(lowest_month['revenue'])}</b>. "
            "The relatively narrow range suggests a stable sales "
            "pattern without major seasonal fluctuations.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # --------------------------------------
    # Revenue by Category
    # --------------------------------------

    story.append(
        Paragraph(
            "Revenue by Category",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["revenue_by_category"],
            width=165 * mm,
            height=82 * mm
        )
    )

    revenue_by_category = analysis["revenue_by_category"]

    top_category = revenue_by_category.iloc[0]

    story.append(
        Paragraph(
            f"<b>Key observation:</b> "
            f"<b>{top_category['category']}</b> generated the highest "
            f"revenue at <b>{format_value(top_category['revenue'])}</b>, "
            "making it the largest contributor to overall sales. "
            "Category performance should therefore be monitored "
            "separately when evaluating product mix and future growth.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        PageBreak()
    )
    
def create_product_analysis_section(story, styles, chart_paths, analysis):
    """Create the product and category analysis section."""

    story.append(
        Paragraph(
            "4. Product & Category Analysis",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Product-level analysis highlights the difference between "
            "sales volume and revenue contribution. This distinction "
            "is important when evaluating product performance and "
            "prioritizing inventory and sales activities.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Top Products by Revenue
    # --------------------------------------

    story.append(
        Paragraph(
            "Top 10 Products by Revenue",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["top_products_revenue"],
            width=165 * mm,
            height=82 * mm
        )
    )

    revenue_products = analysis["revenue_by_product"]

    top_revenue_product = revenue_products.iloc[0]

    story.append(
        Paragraph(
            f"<b>Key observation:</b> "
            f"<b>{top_revenue_product['product']}</b> generated the "
            f"highest product revenue at "
            f"<b>{format_value(top_revenue_product['revenue'])}</b>. "
            f"This product accounted for "
            f"{top_revenue_product['orders']:,} orders and "
            f"{top_revenue_product['units_sold']:,} units sold.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 2 * mm)
    )

    # --------------------------------------
    # Top Products by Units Sold
    # --------------------------------------

    story.append(
        Paragraph(
            "Top 10 Products by Units Sold",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["top_products_units"],
            width=165 * mm,
            height=82 * mm
        )
    )

    volume_products = analysis["top_products_units"]

    top_volume_product = volume_products.iloc[0]

    story.append(
        Paragraph(
            f"<b>Key observation:</b> "
            f"<b>{top_volume_product['product']}</b> had the highest "
            f"sales volume with "
            f"<b>{top_volume_product['units_sold']:,} units sold</b>. "
            f"Its revenue contribution was "
            f"<b>{format_value(top_volume_product['revenue'])}</b>. "
            "The difference between volume and revenue performance "
            "shows why both indicators should be monitored together.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        PageBreak()
    )
    
def create_customer_analysis_section(story, styles, chart_paths, analysis):
    """Create the customer analysis section."""

    story.append(
        Paragraph(
            "5. Customer Analysis",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Customer-level analysis identifies customers with the "
            "highest total revenue and customers with the highest "
            "average order value. These measures provide different "
            "perspectives on customer contribution.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Top Customers by Revenue
    # --------------------------------------

    story.append(
        Paragraph(
            "Top 10 Customers by Revenue",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["top_customers_revenue"],
            width=165 * mm,
            height=82 * mm
        )
    )

    customer_analysis = analysis["customer_analysis"]

    top_revenue_customer = customer_analysis.iloc[0]

    story.append(
        Paragraph(
            f"<b>Key observation:</b> Customer "
            f"<b>{top_revenue_customer['customer_id']}</b> generated "
            f"<b>{format_value(top_revenue_customer['revenue'])}</b> "
            f"in revenue across "
            f"<b>{top_revenue_customer['orders']:,} orders</b>. "
            "Customers with consistently high revenue contribution "
            "represent an important group for retention and "
            "relationship management.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # --------------------------------------
    # Top Customers by AOV
    # --------------------------------------

    story.append(
        Paragraph(
            "Top 10 Customers by Average Order Value",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["top_customers_aov"],
            width=165 * mm,
            height=82 * mm
        )
    )

    top_aov_customer = (
        customer_analysis
        .sort_values(
            "avg_order_value",
            ascending=False
        )
        .iloc[0]
    )

    story.append(
        Paragraph(
            f"<b>Key observation:</b> Customer "
            f"<b>{top_aov_customer['customer_id']}</b> recorded the "
            f"highest average order value at "
            f"<b>{top_aov_customer['avg_order_value']:,.2f} KM</b>. "
            "High-AOV customers may provide opportunities for "
            "premium product offers, bundles, or targeted upselling.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            "<b>Business implication:</b> Total customer revenue and "
            "average order value should be monitored together. A "
            "customer can generate substantial total revenue through "
            "frequent purchases while another customer may generate "
            "higher-value individual orders.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        PageBreak()
    )

def create_geographic_analysis_section(story, styles, chart_paths, analysis):
    """Create the geographic sales analysis section."""

    story.append(
        Paragraph(
            "6. Geographic Analysis",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Sales performance varies across cities. "
            "The analysis compares total revenue, order volume, "
            "and average order value to identify differences "
            "between market size and customer spending patterns.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Revenue by City
    # --------------------------------------

    story.append(
        Paragraph(
            "Revenue by City",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["revenue_by_city"],
            width=165 * mm,
            height=82 * mm
        )
    )

    city_analysis = analysis["revenue_by_city"]

    top_city = city_analysis.iloc[0]

    story.append(
        Paragraph(
            f"<b>{top_city['city']}</b> generated the highest total revenue "
            f"with <b>{top_city['revenue']:,.2f} KM</b>. "
            f"This reflects its higher order volume and overall market activity.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # City AOV Analysis
    # --------------------------------------

    city_by_aov = city_analysis.copy()

    city_by_aov["average_order_value"] = (
        city_by_aov["revenue"] / city_by_aov["orders"]
    )

    city_by_aov = city_by_aov.sort_values(
        "average_order_value",
        ascending=False
    )

    top_aov_city = city_by_aov.iloc[0]

    story.append(
        Paragraph(
            f"<b>{top_aov_city['city']}</b> recorded the highest average "
            f"order value at <b>{top_aov_city['average_order_value']:,.2f} KM</b>. "
            f"This shows that the city with the highest total revenue "
            f"does not necessarily have the highest average customer spend.",
            styles["BodyTextCustom"]
        )
    )

    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "<b>Business implication:</b> Geographic performance should "
            "be evaluated using both total revenue and average order value. "
            "High-revenue markets can indicate strong sales volume, while "
            "higher AOV markets may provide opportunities for increasing "
            "customer value through targeted offers and product combinations.",
            styles["BodyTextCustom"]
        )
    )

    story.append(PageBreak())

def create_discount_payment_analysis_section(
    story,
    styles,
    chart_paths,
    analysis
):
    """Create the discount and payment analysis section."""

    story.append(
        Paragraph(
            "7. Discount & Payment Analysis",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Payment methods and discount levels provide additional "
            "insight into customer purchasing behavior and revenue "
            "generation. The analysis compares revenue contribution "
            "across payment methods and discount levels.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Payment Method Analysis
    # --------------------------------------

    story.append(
        Paragraph(
            "Revenue by Payment Method",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["revenue_by_payment_method"],
            width=165 * mm,
            height=82 * mm
        )
    )

    payment_analysis = analysis["payment_analysis"]

    top_payment = payment_analysis.iloc[0]

    story.append(
        Paragraph(
            f"<b>{top_payment['payment_method']}</b> generated the highest "
            f"revenue contribution with <b>{top_payment['revenue']:,.2f} KM</b>. "
            f"This indicates that this payment method represents a significant "
            f"part of overall transaction activity.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Discount Analysis
    # --------------------------------------

    story.append(
        Paragraph(
            "Revenue by Discount Level",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Image(
            chart_paths["revenue_by_discount"],
            width=165 * mm,
            height=82 * mm
        )
    )

    discount_analysis = analysis["discount_analysis"]

    no_discount = discount_analysis[
        discount_analysis["discount"] == 0
    ].iloc[0]

    story.append(
        Paragraph(
            f"Orders without a discount generated "
            f"<b>{no_discount['revenue']:,.2f} KM</b> in revenue. "
            f"The analysis also shows that average order value decreases "
            f"as discount levels increase.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Paragraph(
            "<b>Business implication:</b> Discounts should be monitored "
            "carefully to ensure that they support sales objectives without "
            "unnecessarily reducing average order value. Payment method "
            "patterns can also help inform customer experience and payment "
            "optimization decisions.",
            styles["BodyTextCustom"]
        )
    )

    story.append(PageBreak())
    
def create_business_insights_section(story, styles, kpis, analysis):
    """Create the business insights section."""

    story.append(
        Paragraph(
            "8. Key Business Insights",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "The analysis identifies several patterns that are relevant "
            "for monitoring sales performance and supporting future "
            "business decisions.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Insight 1 — Stable Sales Performance
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>1. Stable overall sales performance</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            f"Northstar Retail generated <b>{kpis['total_revenue']:,.2f} KM</b> "
            f"from <b>{kpis['total_orders']:,}</b> orders during 2025. "
            "Monthly revenue remained relatively stable throughout the year, "
            "with no extreme concentration of annual revenue in a single month.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Insight 2 — Electronics
    # --------------------------------------

    top_category = analysis["revenue_by_category"].iloc[0]

    story.append(
        Paragraph(
            "<b>2. Electronics is the leading revenue category</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            f"The <b>{top_category['category']}</b> category generated "
            f"<b>{top_category['revenue']:,.2f} KM</b>, making it the "
            "largest contributor to total revenue. This category should "
            "therefore remain an important focus of product and sales monitoring.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Insight 3 — Revenue vs Volume
    # --------------------------------------

    top_revenue_product = analysis["revenue_by_product"].iloc[0]

    story.append(
        Paragraph(
            "<b>3. Revenue contribution and sales volume are different</b>",
            styles["SubsectionTitle"]
        )   
    )

    story.append(
        Paragraph(
            f"<b>{top_revenue_product['product']}</b> generated the highest "
            f"product revenue at <b>{top_revenue_product['revenue']:,.2f} KM</b>. "
            "At the same time, Wireless Mouse recorded the highest unit volume "
            "with 5,169 units sold. This demonstrates why both revenue and "
            "sales volume should be monitored when evaluating product performance.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Insight 4 — Geographic Performance
    # --------------------------------------

    top_city = analysis["revenue_by_city"].iloc[0]

    story.append(
        Paragraph(
            "<b>4. Geographic revenue is driven partly by order volume</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            f"<b>{top_city['city']}</b> generated the highest total revenue "
            f"at <b>{top_city['revenue']:,.2f} KM</b>. "
            "Its position is associated with a higher number of orders, "
            "rather than having the highest average order value. "
            "This distinction is important when comparing markets.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Insight 5 — Discounts
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>5. Higher discount levels are associated with lower AOV</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "The analysis shows that average order value decreases as "
            "discount levels increase. This is an observed association "
            "within the dataset and should not be interpreted as proof "
            "that discounts directly cause lower order values.",
            styles["BodyTextCustom"]
        )
    )

    story.append(PageBreak())

def create_recommendations_section(story, styles):
    """Create the recommendations section."""

    story.append(
        Paragraph(
            "9. Recommendations",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Based on the observed sales patterns and business metrics, "
            "the following actions can support revenue growth, customer "
            "value, and more consistent performance monitoring.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 1
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>1. Prioritize high-value products</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Maintain strong availability and visibility for products "
            "that generate a significant share of revenue. Product-level "
            "revenue should be monitored regularly to identify changes "
            "in demand and contribution.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 2
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>2. Monitor revenue and sales volume together</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "High unit volume does not necessarily correspond to the "
            "highest revenue contribution. Product decisions should "
            "therefore consider both units sold and revenue generated.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 3
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>3. Focus on increasing Average Order Value</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Use product bundles, complementary products, and targeted "
            "cross-selling opportunities to increase the value of "
            "individual customer orders without relying exclusively "
            "on discounts.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 4
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>4. Use discounts selectively</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Because higher discount levels are associated with lower "
            "average order value in the dataset, discounts should be "
            "targeted toward specific commercial objectives rather than "
            "applied broadly. Their impact should be monitored using "
            "revenue, order volume, and AOV.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 5
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>5. Monitor geographic performance</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Track both revenue and AOV by city. High-volume markets "
            "can be managed for scale, while markets with higher AOV "
            "may provide opportunities for increasing customer value "
            "through targeted campaigns and product combinations.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # Recommendation 6
    # --------------------------------------

    story.append(
        Paragraph(
            "<b>6. Establish regular KPI monitoring</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Create a monthly reporting process covering revenue, "
            "orders, AOV, units sold, category performance, discounts, "
            "and geographic performance. Regular monitoring will make "
            "it easier to identify changes early and evaluate the impact "
            "of business actions.",
            styles["BodyTextCustom"]
        )
    )

    story.append(PageBreak())

def create_kpi_targets_section(story, styles, kpis):
    """Create KPI targets and next steps section."""

    story.append(
        Paragraph(
            "10. KPI Targets & Next Steps",
            styles["SectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "The following targets are proposed management objectives "
            "for future performance monitoring. They are not forecasts "
            "and are not directly derived from the historical dataset.",
            styles["BodyTextCustom"]
        )
    )

    # --------------------------------------
    # KPI Targets Table
    # --------------------------------------

    kpi_target_data = [
        [
            Paragraph("<b>KPI</b>", styles["SmallText"]),
            Paragraph("<b>2025 Result</b>", styles["SmallText"]),
            Paragraph("<b>Proposed Target</b>", styles["SmallText"]),
        ],
        [
            "Total Revenue",
            f"{kpis['total_revenue']:,.2f} KM",
            "3.33M KM (+10%)"
        ],
        [
            "Total Orders",
            f"{kpis['total_orders']:,}",
            "21,000+"
        ],
        [
            "Average Order Value",
            f"{kpis['average_order_value']:,.2f} KM",
            "160+ KM"
        ],
        [
            "Units Sold",
            f"{kpis['units_sold']:,}",
            "37,000+"
        ],
        [
            "Electronics Revenue",
            "1.21M KM",
            "+10%"
        ],
        [
            "Discounted Revenue Share",
            "52.28%",
            "≤ 50%"
        ],
        [
            "Monthly Revenue",
            "234.6K–272.7K KM",
            "≥ 250K KM"
        ],
        [
            "Critical Data Quality Issues",
            "Multiple issues identified",
            "< 1%"
        ],
    ]

    kpi_table = Table(
        kpi_target_data,
        colWidths=[55 * mm, 55 * mm, 55 * mm]
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D9E2F3")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.HexColor(MAIN_COLOR)
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor(MAIN_COLOR)
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#B4C7E7")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(kpi_table)

    story.append(Spacer(1, 8 * mm))

    # --------------------------------------
    # Next Steps
    # --------------------------------------

    story.append(
        Paragraph(
            "Recommended Next Steps",
            styles["SubsectionTitle"]
        )
    )

    next_steps = [
        "Establish a monthly sales KPI review using the metrics in this report.",
        "Monitor product revenue and unit volume to identify changes in product performance.",
        "Review discount performance regularly and compare discount levels with AOV and revenue.",
        "Track city-level revenue and AOV to identify opportunities for market development.",
        "Improve data-quality controls for critical transaction fields.",
        "Use the 2025 results as a baseline for evaluating future performance."
    ]

    for step in next_steps:
        story.append(
            Paragraph(
                f"• {step}",
                styles["BodyTextCustom"]
            )
        )

    story.append(PageBreak())
    
def create_final_conclusion_section(story, styles, kpis, analysis):
    """Create the final conclusion section."""

    story.append(
        Paragraph(
            "11. Final Conclusion",
            styles["SectionTitle"]
        )
    )

    top_category = analysis["revenue_by_category"].iloc[0]
    top_city = analysis["revenue_by_city"].iloc[0]

    story.append(
        Paragraph(
            f"Northstar Retail generated "
            f"<b>{kpis['total_revenue']:,.2f} KM</b> in revenue from "
            f"<b>{kpis['total_orders']:,}</b> orders during 2025, with an "
            f"average order value of <b>{kpis['average_order_value']:,.2f} KM</b>.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Paragraph(
            f"The analysis identified <b>{top_category['category']}</b> "
            f"as the largest revenue-generating category, while "
            f"<b>{top_city['city']}</b> generated the highest total "
            f"revenue among the analyzed cities.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Paragraph(
            "The results indicate that future performance monitoring "
            "should focus on revenue growth, Average Order Value, "
            "product-level performance, geographic differences, "
            "and disciplined use of discounts.",
            styles["BodyTextCustom"]
        )
    )

    story.append(
        Paragraph(
            "A structured monthly KPI review, combined with improved "
            "data-quality monitoring, can provide a consistent basis "
            "for evaluating business performance and measuring progress "
            "against the proposed management targets.",
            styles["BodyTextCustom"]
        )
    )

    story.append(Spacer(1, 15 * mm))

    story.append(
        Paragraph(
            "<b>End of Report</b>",
            styles["SubsectionTitle"]
        )
    )

    story.append(
        Paragraph(
            "Northstar Retail — Business Sales Analysis 2025",
            styles["SmallText"]
        )
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
        
    generate_pdf(
        kpis,
        analysis,
        chart_paths
        )