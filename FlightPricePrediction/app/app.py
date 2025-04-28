from flask import Flask, request, render_template
import joblib
import datetime
import matplotlib.pyplot as plt
import os
import holidays
import numpy as np
import pandas as pd
from utils.db_connection import get_flight_data

app = Flask(__name__)
fare_classes = ["Economy", "PremiumEconomy", "Business", "First"]
us_holidays = holidays.US()

def holiday_flag(date):
    return 1 if date in us_holidays else 0

def get_average_fare(origin, destination, fare_class, month):
    df = get_flight_data()
    df = df[(df["Origin"] == origin) & (df["Destination"] == destination)]
    df = df[pd.to_datetime(df["Departure_Date"]).dt.month == month]
    return df[fare_class].mean()

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        origin = request.form["origin"]
        destination = request.form["destination"]
        price = float(request.form["price"])
        booking_date = request.form["booking_date"]
        travel_date = request.form["travel_date"]
        fare_class = request.form["fare_class"]

        basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(basedir, "model", f"{fare_class}_model.pkl")

        print(f"Looking for model at: {model_path}")
        if not os.path.exists(model_path):
            return f"Model for {fare_class} not trained. Checked: {model_path}"

        model = joblib.load(model_path)

        booking_date = datetime.datetime.strptime(booking_date, "%Y-%m-%d")
        travel_date = datetime.datetime.strptime(travel_date, "%Y-%m-%d")
        days_until_departure = (travel_date - booking_date).days

        forecast_offsets = [0, 15, 45, 60, 90]
        prediction_days = [days_until_departure + offset for offset in forecast_offsets]

        features = [[
            d,
            holiday_flag(booking_date + datetime.timedelta(days=d)),
            (booking_date + datetime.timedelta(days=d)).month,
            (booking_date + datetime.timedelta(days=d)).year,
            (booking_date + datetime.timedelta(days=d)).weekday()
        ] for d in prediction_days]

        feature_df = pd.DataFrame(features, columns=[
            "days_until_departure", "holiday_flag", "month", "year", "day_of_week"
        ])

        predicted_increases = model.predict(feature_df)

        avg_base_fare = get_average_fare(origin, destination, fare_class, travel_date.month)
        if pd.isna(avg_base_fare):
            return "❌ Could not determine average base fare for this route/month. Please try another."

        forecast_prices = [round(avg_base_fare * (1 + pct), 2) for pct in predicted_increases]
        forecast_dict = dict(zip([15, 45, 60, 90], forecast_prices[1:]))

        optimal_price = min(forecast_prices)
        deal = "Yes" if price < optimal_price else "No"

        plot_filename = f"forecast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
        plot_path = os.path.join("static", plot_filename)
        os.makedirs("static", exist_ok=True)

        plt.figure()
        plt.plot(prediction_days, forecast_prices, marker="o")
        plt.title(f"{fare_class} Optimal Price Forecast")
        plt.xlabel("Days Before Departure")
        plt.ylabel("Predicted Optimal Price")
        plt.savefig(plot_path)
        plt.close()

        return render_template(
            "result.html",
            is_good_deal=deal,
            forecast=forecast_dict,
            plot_path=plot_path,
            fare_class=fare_class,
            optimal_price=round(optimal_price, 2),
            input_price=price
        )

    return render_template("form.html", fare_classes=fare_classes)

if __name__ == '__main__':
    app.run(debug=True)