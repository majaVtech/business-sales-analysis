# Business Sales Analysis

## Project Overview

This project presents a complete business sales analysis for **Northstar Retail**, a fictional retail company.

The goal of the project is to evaluate sales performance, identify key business trends, analyze customer and product performance, and translate the results into actionable business insights and recommendations.

The analysis is based on **2025 sales transaction data** and follows a practical business analytics workflow from data quality assessment and cleaning to KPI analysis, visualization, and reporting.

## Business Objectives

The analysis focuses on:

* Evaluating overall sales performance
* Tracking key business KPIs
* Identifying top-performing products and categories
* Analyzing customer purchasing behavior
* Comparing geographic sales performance
* Evaluating payment methods
* Examining discount usage and its relationship with order value
* Identifying actionable business opportunities

## Key Results

The analysis identified the following key results for 2025:

* **Total Revenue:** 3.03M KM
* **Total Orders:** 19,985
* **Units Sold:** 35,064
* **Average Order Value:** 151.58 KM
* **Average Revenue per Unit:** 86.39 KM
* **Top Revenue Category:** Electronics
* **Top Revenue Product:** Smartphone
* **Highest Revenue City:** Banja Luka

The analysis also found that higher discount levels were associated with lower average order values. This relationship is treated as an observed business pattern rather than a causal effect.

## Data Preparation

The raw dataset contained several data-quality issues, including:

* Duplicate transactions
* Missing city values
* Missing payment methods
* Inconsistent category capitalization
* Negative quantities
* Negative unit prices

The data was cleaned before analysis by removing invalid transactions, eliminating complete duplicates, standardizing categorical values, and converting dates to the appropriate format.

Missing city and payment-method values were retained because the corresponding transactions remained valid for other parts of the analysis.

## Analysis Performed

The project includes analysis of:

* Monthly revenue
* Revenue by category
* Product performance
* Customer performance
* Geographic performance
* Payment methods
* Discount levels
* Key business KPIs

The project also includes business recommendations and measurable KPI targets for future performance monitoring.

## Project Structure

```text
business-sales-analysis/
│
├── Data/
│   └── raw_sales_data.csv
│
├── Notebooks/
│   └── 01_sales_analysis.ipynb
│
├── Reports/
│   ├── generate_report.py
│   └── charts/
│
├── src/
│
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook
* ReportLab
* Git & GitHub

## Business Report

A professional PDF report is generated using the `generate_report.py` script.

The report includes:

* Executive summary
* Business KPIs
* Sales performance analysis
* Product and category analysis
* Customer analysis
* Geographic analysis
* Payment and discount analysis
* Business insights
* Recommendations
* KPI targets and next steps

## Purpose

This project demonstrates how business transaction data can be transformed into clear, decision-oriented insights using Python and data analytics.

It is designed as a portfolio example of a practical **business data analysis and reporting workflow**.
