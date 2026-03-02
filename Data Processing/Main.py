from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "FilteredV1.csv"
OUTPUT_PATH = BASE_DIR / "FilteredV1_filled.csv"
PRICE_COLUMN = "Price"
TYPE_COLUMN = "Type"

def replace_zero_prices(input_df: pd.DataFrame) -> pd.DataFrame:
    prices = pd.to_numeric(input_df[PRICE_COLUMN], errors="coerce")
    avg_by_type = (
        input_df.assign(_price=prices)
        .replace({"_price": {0: pd.NA}})
        .groupby(TYPE_COLUMN)["_price"]
        .mean()
    )
    zero_mask = prices == 0
    fill_values = input_df[TYPE_COLUMN].map(avg_by_type).astype(float).round()
    filled_mask = zero_mask & fill_values.notna()
    output_df = input_df.copy()
    output_df[PRICE_COLUMN] = pd.to_numeric(output_df[PRICE_COLUMN], errors="coerce").astype(float)
    output_df.loc[filled_mask, PRICE_COLUMN] = fill_values[filled_mask]
    return output_df

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    updated_df = replace_zero_prices(df)
    updated_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()