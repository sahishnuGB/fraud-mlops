from src.validate import validate_frame


def test_valid_batch_has_no_errors(batch_df, params):
    assert validate_frame(batch_df, params["validation"]) == []


def test_missing_column(batch_df, params):
    errors = validate_frame(batch_df.drop(columns=["amount"]), params["validation"])
    assert errors and "missing columns" in errors[0]


def test_bad_values_are_reported(batch_df, params):
    df = batch_df.copy()
    df.loc[0, "amount"] = -5.0
    df.loc[1, "merchant_category"] = "casino"
    df.loc[2, "is_fraud"] = 3
    df.loc[3, "transaction_id"] = df.loc[4, "transaction_id"]
    text = " | ".join(validate_frame(df, params["validation"]))
    assert "amount: negative" in text
    assert "casino" in text
    assert "is_fraud" in text
    assert "duplicate transaction_id" in text


def test_nulls_and_row_count(batch_df, params):
    df = batch_df.head(50).copy()
    df["customer_age"] = df["customer_age"].astype("float")
    df.loc[0, "customer_age"] = None
    text = " | ".join(validate_frame(df, params["validation"]))
    assert "only 50 rows" in text
    assert "customer_age: 1 null" in text
