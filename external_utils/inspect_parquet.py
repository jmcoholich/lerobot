import sys
import pandas as pd

def inspect_parquet(path, num_rows=5):
    print(f"\n=== Loading: {path} ===\n")

    df = pd.read_parquet(path)

    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(list(df.columns))

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== SAMPLE ROWS ===")
    print(df.head(num_rows))

    print("\n=== FIRST ROW (FULL) ===")
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        print(f"\n--- {col} ---")
        print(type(val))
        try:
            print(val if len(str(val)) < 500 else str(val)[:500] + "...")
        except:
            print(val)
    breakpoint()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <file.parquet>")
        sys.exit(1)

    inspect_parquet(sys.argv[1])