from rich import print
import pandas as pd
import os

# Input and output
data_chunks = pd.read_csv(
    r"c:\Users\ghazkul\Downloads\jena_climate_2009_2016.csv", chunksize=4096
)
output_file_path = "formated.csv"

# Remove the file if it exists (optional: clean start)
if os.path.exists(output_file_path):
    os.remove(output_file_path)

write_header = True  # Initialize header flag

for chunk in data_chunks:
    if "T (degC)" in chunk.columns:
        num_rows = len(chunk)
        records = []

        for i in range(0, num_rows - 4, 5):
            record = {
                "data_1": chunk["T (degC)"].iloc[i],
                "data_2": chunk["T (degC)"].iloc[i + 1],
                "data_3": chunk["T (degC)"].iloc[i + 2],
                "data_4": chunk["T (degC)"].iloc[i + 3],
                "label": chunk["T (degC)"].iloc[i + 4],
            }
            records.append(record)

        if records:
            df_to_write = pd.DataFrame(records)
            df_to_write.to_csv(
                output_file_path, mode="a", header=write_header, index=False
            )
            write_header = False
    else:
        print(
            "[yellow]Warning: 'T (degC)' column not found in this chunk. Skipping chunk.[/yellow]"
        )

print(
    f"[green]Successfully wrote the five temperature columns to {output_file_path}[/green]"
)
