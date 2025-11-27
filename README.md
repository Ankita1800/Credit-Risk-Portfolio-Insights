# Credit Risk Portfolio Insights

A comprehensive, single-file Python solution for analyzing and modeling credit risk in a loan portfolio using Lending Club data.

## Overview

This project provides an end-to-end credit risk analysis pipeline that:
- **Loads & Preprocesses** Lending Club loan data (100K+ loans)
- **Engineers Features** with domain-specific transformations
- **Develops PD Model** using logistic regression
- **Performs Portfolio Analytics** with risk segmentation
- **Generates Visualizations** and detailed reports

## Project Structure

```
credit-risk-portfolio-insights/
├─ credit_risk_portfolio_insights.py   # Complete analysis pipeline (single file)
├─ accepted_2007_to_2018Q4.csv         # Accepted loans data (Lending Club)
├─ rejected_2007_to_2018Q4.csv         # Rejected applications data
├─ README.md                           # This file
└─ requirements.txt                    # Python dependencies
```

## Features

### 1. Data Loading & Preprocessing
- Loads Lending Club dataset with 100K+ loans and 150+ features
- Creates binary default target variable from loan_status
- Filters to completed loans (Fully Paid or Charged Off)

### 2. Feature Engineering
- Cleans and transforms 17 key credit risk features
- Encodes categorical variables (grade, home_ownership, verification_status)
- Converts employment length to numeric values
- Creates FICO score average and debt-to-income ratio handling

### 3. Probability of Default (PD) Modeling
- Trains logistic regression model on 80% of data
- Evaluates on 20% test set
- Reports: AUC-ROC, F1-Score, confusion matrix, classification report
- Generates PD predictions for entire portfolio

### 4. Portfolio Analytics
- Calculates key risk metrics (weighted average PD, concentration, etc.)
- Segments portfolio into 5 risk tiers (Very Low to Very High)
- Computes expected losses (PD × Exposure × LGD)
- Analyzes risk distribution across segments

### 5. Visualization & Reporting
- **Dashboard**: Multi-panel visualization with PD distribution, ROC curve, risk segments
- **CSV Output**: Full portfolio with PD predictions and risk labels
- **Text Report**: Comprehensive summary of all metrics and findings

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Navigate to project directory:
```bash
cd credit-risk-portfolio-insights
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis pipeline with a single command:

```bash
python credit_risk_portfolio_insights.py
```

### Default Configuration
- **Data**: First 100,000 loans from `accepted_2007_to_2018Q4.csv`
- **Model**: Logistic Regression with StandardScaler normalization
- **LGD**: 45% (Loss Given Default assumption)
- **Train-Test Split**: 80-20

### Customization

Edit the `main()` function in `credit_risk_portfolio_insights.py`:

```python
# Change data sample size (adjust nrows parameter)
loader = DataLoader('accepted_2007_to_2018Q4.csv', nrows=50000)  # Use 50K instead of 100K

# Change LGD assumption (adjust lgd parameter)
portfolio.calculate_expected_loss(lgd=0.50)  # Use 50% LGD instead of 45%

# Change number of risk segments (adjust n_segments parameter)
portfolio.segment_portfolio(n_segments=10)  # Use 10 segments instead of 5
```

## Output Files

After running the script, you'll get:

1. **portfolio_analysis_results.csv** - Complete portfolio with predictions and risk labels
   - Columns: id, loan_amnt, int_rate, grade, loan_status, default, predicted_pd, risk_label, expected_loss

2. **portfolio_analysis_dashboard.png** - Comprehensive visualization dashboard with:
   - Portfolio PD distribution
   - ROC curve for model evaluation
   - Risk segment distribution (by count and exposure)
   - Expected loss breakdown
   - Top 10 feature importance

3. **portfolio_summary_report.txt** - Text report with all metrics and findings

## Key Metrics Explained

| Metric | Definition |
|--------|-----------|
| **PD (Probability of Default)** | Likelihood a borrower will default (0-1) |
| **Weighted Avg PD** | PD weighted by loan amounts across portfolio |
| **Expected Loss** | PD × Loan Amount × LGD (typical reserve needed) |
| **AUC-ROC** | Model discrimination ability (0.5=random, 1.0=perfect) |
| **LGD (Loss Given Default)** | Assumption of % loss if default occurs (here 45%) |

## Features Used in PD Model

1. loan_amnt - Loan principal amount
2. int_rate - Interest rate
3. installment - Monthly payment amount
4. grade_num - Loan grade (A-G converted to 1-7)
5. annual_inc - Annual income
6. dti - Debt-to-income ratio
7. fico_score - Average FICO credit score
8. open_acc - Number of open credit accounts
9. revol_bal - Revolving balance
10. revol_util - Revolving credit utilization rate
11. total_acc - Total credit accounts
12. emp_length_num - Employment length in years
13. home_ownership_num - Home ownership status (encoded)
14. verification_num - Verification status (encoded)
15. term_months - Loan term in months
16. pub_rec - Public records/derogatory marks
17. inq_last_6mths - Credit inquiries in last 6 months

## Example Output

```
================================================================================
CREDIT RISK PORTFOLIO INSIGHTS - COMPREHENSIVE ANALYSIS
================================================================================

[1] LOADING DATA
────────────────────────────────────────────────────────────────────────────────
✓ Loaded 100,000 rows with 151 columns
✓ Loan Status Distribution:
   Fully Paid    50,000
   Charged Off   47,000
   ...

[8] PORTFOLIO RISK METRICS
────────────────────────────────────────────────────────────────────────────────
✓ Total Exposure: $2,543,287,654.23
✓ Number of Loans: 100,000
✓ Average PD: 0.1845 (18.45%)
✓ Weighted Avg PD: 0.1623 (16.23%)
✓ Total Expected Loss (LGD=45%): $185,637,842.34

[9] PORTFOLIO SEGMENTATION
────────────────────────────────────────────────────────────────────────────────
   Very Low    : 20,000 loans | $512,456,789 (20.1%) | 3.21% PD
   Low         : 20,000 loans | $506,234,567 (19.9%) | 8.45% PD
   Medium      : 20,000 loans | $510,123,456 (20.0%) | 15.63% PD
   High        : 20,000 loans | $508,987,654 (20.0%) | 25.34% PD
   Very High   : 20,000 loans | $505,485,188 (19.9%) | 38.92% PD

✅ ANALYSIS COMPLETE!
```

## Dependencies

```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=0.24.0   # Machine learning
matplotlib>=3.4.0      # Visualization
seaborn>=0.11.0        # Statistical visualization
```

## Performance

- **Processing Time**: ~30-60 seconds for 100K loans (varies by machine)
- **Memory Usage**: ~2-3 GB for 100K loans dataset
- **Output Size**: ~15-20 MB for CSV + PNG files

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: Install scikit-learn
```bash
pip install scikit-learn
```

### Issue: "FileNotFoundError: accepted_2007_to_2018Q4.csv not found"
**Solution**: Ensure the CSV file is in the same directory as the script

### Issue: "MemoryError" on large datasets
**Solution**: Reduce nrows parameter in main() function

## Next Steps

After running the analysis:
1. Review `portfolio_summary_report.txt` for key findings
2. Check `portfolio_analysis_dashboard.png` for visualizations
3. Import `portfolio_analysis_results.csv` into your BI tool for dashboarding
4. Use predicted_pd column for risk-based pricing or portfolio decisions

## License

Proprietary - All rights reserved

## Contact

Credit Risk Analytics Team
November 27, 2025
