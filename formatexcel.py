import pandas as pd
import json
import openpyxl

# Load the Excel file
df = pd.read_excel('HNMV_DLinear.xlsx')

# Use json.loads to convert the 'metrics' column string representations to actual lists
df['metrics'] = df['metrics'].apply(json.loads)

# Convert these lists into separate columns
metrics_df = pd.DataFrame(df['metrics'].to_list(), columns=['metric1', 'metric2', 'metric3', 'metric4', 'metric5'])

# Concatenate the new columns with the original DataFrame
df = pd.concat([df.drop(columns=['metrics']), metrics_df], axis=1)

# Keep only metric1 and metric2
df = df.drop(columns=['metric3', 'metric4', 'metric5'])

# Define the columns to group by (first 20 columns)
group_columns = df.columns[:18].tolist()

# Group by the first 20 columns and keep the row with the best metric1
df = df.sort_values('metric2', ascending=True).groupby(group_columns, as_index=False).first()

# Save the updated DataFrame to a new Excel file
output_file = 'HNMV_DLinear_optimized.xlsx'
df.to_excel(output_file, index=False)

# Open the saved file and set number format for all cells
wb = openpyxl.load_workbook(output_file)
ws = wb.active

for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
    for cell in row:
        if isinstance(cell.value, float):
            cell.number_format = '0.000000'  # Format to show 6 decimal places

# Save the workbook
wb.save(output_file)