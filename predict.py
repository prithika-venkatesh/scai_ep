import joblib

def predict_purchase_amount(item, payment_method, review_rating, purchase_month, purchase_day, purchase_weekday):
    model = joblib.load('purchase_amount_model.pkl')
    input_data = {
        'Item Purchased': [item],
        'Payment Method': [payment_method],
        'Review Rating': [review_rating],
        'Purchase Month': [purchase_month],
        'Purchase Day': [purchase_day],
        'Purchase Weekday': [purchase_weekday]
    }
    
    import pandas as pd
    df = pd.DataFrame(input_data)
    prediction = model.predict(df)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    # Example inputs
    item = input("Enter Item Purchased (e.g., Handbag): ")
    payment_method = input("Enter Payment Method (Credit Card or Cash): ")
    review_rating = float(input("Enter Review Rating (0–5): "))
    purchase_month = int(input("Enter Purchase Month (1–12): "))
    purchase_day = int(input("Enter Purchase Day (1–31): "))
    purchase_weekday = int(input("Enter Purchase Weekday (0=Mon, ..., 6=Sun): "))

    predicted_amount = predict_purchase_amount(
        item, payment_method, review_rating, purchase_month, purchase_day, purchase_weekday
    )

    print(f"Predicted Purchase Amount: ${predicted_amount}")
